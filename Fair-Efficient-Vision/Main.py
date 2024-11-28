#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import models_vit
from timm.models.layers import trunc_normal_
from pos_embed import interpolate_pos_embed
from Dataset import build_combined_dataset
import lr_decay as lrd
import pandas as pd
from NativeScalar import NativeScalerWithGradNormCount as NativeScaler
import lr_sched as lr_sched
from sklearn.preprocessing import label_binarize

# In[2]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Data_Image_size = 200
global_pool=True
num_classes=3


# In[3]:


def load_model(resume,evaluation, model_without_ddp, optimizer,loss_scaler):
    if resume:
        if resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (evaluation):
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


# In[4]:


def RetFound_TeacherModel():
    model = models_vit.__dict__['vit_large_patch16'](
            img_size=200,
            num_classes=3,
            drop_path_rate=0.1,
            global_pool=True,
        )
    checkpoint_path = './checkpoint-best.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("Load pre-trained checkpoint from: %s" % checkpoint_path)
    
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    
    for k in ['head.weight', 'head.bias']:
        print(checkpoint_model[k].shape)
        print(state_dict[k].shape)
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
            
    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)
    
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    global_pool = True

    trunc_normal_(model.head.weight, std=2e-5)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params (M): %.2f' % (n_parameters / 1.e6))
    
    
    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))
    logging.info(f"Teacher Model --> No. of Parameter :  {n_parameters / 1.e6}")

    param_groups = lrd.param_groups_lrd(model_without_ddp, 0.05,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=0.65
    )
    optimizer = torch.optim.AdamW(param_groups,weight_decay=0.02, lr=0.001)
    loss_scaler = NativeScaler()
    load_model(False,False, model_without_ddp=model_without_ddp, optimizer=optimizer,loss_scaler = loss_scaler)
    
    return model,optimizer,loss_scaler


# In[5]:


def build_combined_dataloaders(root_dirs, image_size, batch_size = 40 , num_workers = 10):
    """
    Build DataLoaders for training, validation and testing from two directories.

    Args:
      root_dirs: List of two directory paths.
      args: Arguments containing batch size and other parameters.

    Returns:
      train_loader: DataLoader for training set.
      val_loader: DataLoader for validation set.
      test_loader: DataLoader for test set.
    """
    
    # Training Dataset and DataLoader
    train_dataset = build_combined_dataset(root_dirs=root_dirs, phase='train', input_size=image_size)
    
    train_sampler = SequentialSampler(train_dataset)  
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,  shuffle=True,
    )
    
    # Validation Dataset and DataLoader
    val_dataset = build_combined_dataset(root_dirs=root_dirs, phase='val', input_size=image_size)
    
    val_sampler = SequentialSampler(val_dataset)  
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True, shuffle=True,
    )
    

    test_dataset = build_combined_dataset(root_dirs=root_dirs, phase='test', input_size=image_size)
    
    test_sampler = SequentialSampler(test_dataset)  
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True, shuffle=True,
    )
    logger.info(f"Load the dataloaders for Train Test Validation")
    return train_loader, val_loader, test_loader


# In[6]:


def combined_loss(teacher_output,student_output,teacher_logits,student_logits, true_labels, temperature=2.0, alpha=0.5):
    """
    Combines distillation loss (KL Divergence) with supervised loss (Cross-Entropy).
    
    Args:
        student_logits: Logits from student model.
        teacher_logits: Logits from teacher model.
        true_labels: Ground truth labels.
        temperature: Temperature scaling factor for distillation.
        alpha: Weighting factor between soft and hard losses.
    
    Returns:
        Combined loss value.
    """

    distillation_loss = nn.KLDivLoss(reduction='batchmean')(nn.functional.log_softmax(student_logits / temperature, dim=1),
                                       nn.functional.softmax(teacher_logits / temperature, dim=1)) * (temperature ** 2)
    
    # Cross-Entropy Loss (Supervised) for both teacher and student models
    supervised_loss_teacher = nn.CrossEntropyLoss()(teacher_output, true_labels)
    supervised_loss_student = nn.CrossEntropyLoss()(student_output, true_labels)
    
    # Combine losses with alpha weighting
    total_loss = alpha * distillation_loss + (1 - alpha) * (supervised_loss_student + supervised_loss_teacher)
    
    return total_loss


# In[7]:


def calculate_metrics(y_true, y_pred_probs):
    y_prob = torch.softmax(y_pred_probs, dim=1)
    y_pred = torch.argmax(y_prob, dim=1).cpu().numpy()  
    y_true = y_true.cpu().numpy()                             
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    
    try:
        overall_unique_classes = np.unique(y_true)
        if len(overall_unique_classes) == 2:
            y_score_reduced = y_prob[:, overall_unique_classes[0]]
            overall_auroc = roc_auc_score(y_true, y_score_reduced.detach().numpy())
        if len(overall_unique_classes) >= 3:
            y_true_onehot = label_binarize(y_true, classes=range(3))
            overall_auroc = roc_auc_score(y_true_onehot, y_prob, multi_class='ovr', average='macro')
        else:
            print("Batch contains only one class. Setting AUROC to 0.333")
            overall_auroc = 0.33
    except ValueError:
        auroc = float('nan') 

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'auroc': overall_auroc,
    }



# In[8]:


def train_teacher_student(logger,teacher_model, student_model, optimizer_teacher, optimizer_student,Teacher_loss_scaler, Student__loss_scaler, train_loader,
                          val_loader=None, test_loader=None,
                          epochs=10):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    teacher_model.to(device)   
    student_model.to(device)   
    
    if optimizer_teacher is None or optimizer_student is None:
        logger.warning(f"Optimizer has been intialized. No Layer and weight decay has been applied.")
        optimizer_teacher = optim.AdamW(teacher_model.parameters(), lr=1e-5)   
        optimizer_student = optim.AdamW(student_model.parameters(), lr=1e-4)   

    best_val_accuracy = 0.0 
    best_tech_val_accuracy = 0.0
    accuracy_history = []
    for epoch in range(epochs):
        
        teacher_model.train()  
        student_model.train()   
        
        running_loss = 0.0
        accum_iter = 1
        
        for data_iter_step,(inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            if data_iter_step % accum_iter == 0:
                lr_sched.adjust_learning_rate(optimizer_teacher, data_iter_step / len(train_loader) + epoch, lr = 1e-3)
                lr_sched.adjust_learning_rate(optimizer_student, data_iter_step / len(train_loader) + epoch, lr = 1e-3)
            
            teacher_outputs, teacher_logit = teacher_model(inputs)
            student_outputs, student_logit = student_model(inputs)

            loss = combined_loss(student_outputs, teacher_outputs,teacher_logit,student_logit, labels)
            scaler = torch.cuda.amp.GradScaler()
            scaler.scale(loss).backward(create_graph=False)
            scaler.unscale_(optimizer_teacher) 
            scaler.unscale_(optimizer_student)
            teacher_norm = torch.nn.utils.clip_grad_norm_(teacher_model.parameters(), 3)
            student_norm = torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1)
            scaler.step(optimizer_teacher)
            scaler.step(optimizer_student)
            scaler.update()
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer_teacher.zero_grad()
                optimizer_student.zero_grad()
        

            running_loss += loss.item()

        
        avg_train_loss = running_loss / len(train_loader)   # Average training loss
        
        logger.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}")
        
        if val_loader:
            val_metrics = evaluate_model(student_model, val_loader)
            teacher_metrics = evaluate_model(teacher_model, val_loader)
            writer.add_scalar('perf/val_acc1', val_metrics['accuracy'], epoch)
            writer.add_scalar('perf/val_auc', val_metrics['auroc'], epoch)
            writer.add_scalar('perf/val_loss', avg_train_loss, epoch)
            
            logger.info(f"Student : Validation Metrics at Epoch {epoch+1}: {val_metrics}")
            logger.info(f"Teacher : Validation Metrics at Epoch {epoch+1}: {teacher_metrics}")
            val_metrics['epoch'] = epoch+1
            val_metrics['category'] = 'Validation'
            print(f"Student : Epoch [{epoch+1}/{epochs}], Validation : {val_metrics}")
            print(f"Teacher : Epoch [{epoch+1}/{epochs}], Validation : {teacher_metrics}")
            accuracy_history.append(val_metrics)
            if val_metrics['auroc'] > best_val_accuracy:
                best_val_accuracy = val_metrics['accuracy']
                torch.save(student_model.state_dict(), output_directory + 'best_student_model.pth')  
                torch.save(teacher_model.state_dict(), output_directory + 'best_teacher_model.pth')
                logger.info(f"Best model saved at epoch {epoch+1} with accuracy: {best_val_accuracy:.4f}, Precision : {val_metrics['precision']}, AUROC : {val_metrics['auroc']}")
            if teacher_metrics['auroc'] > best_tech_val_accuracy:
                best_tech_val_accuracy = teacher_metrics['accuracy']
                torch.save(teacher_model.state_dict(), output_directory + 'best_teacher_model.pth')
                logger.info(f"Best model saved at epoch {epoch+1} with accuracy: {best_tech_val_accuracy:.4f}, Precision : {teacher_metrics['precision']}, AUROC : {teacher_metrics['auroc']}")    
            accuracy_df = pd.DataFrame(accuracy_history)
    if test_loader:
        test_metrics = evaluate_model(student_model, test_loader) 
        logger.info(f"Test Metrics: {test_metrics}")
    return accuracy_df


# In[9]:

def evaluate_model(model, dataloader):
    
    model.eval()  
    
    all_labels = []
    all_preds_probs = []

    device = next(model.parameters()).device  
    
    with torch.no_grad(): 
        
        for inputs, labels in dataloader:
            
            inputs = inputs.to(device)
            output,_ = model(inputs)
            outputs_probs = torch.softmax(output, dim=1)   
            
            all_labels.append(labels)
            all_preds_probs.append(outputs_probs.cpu()) 

    
    all_labels = torch.cat(all_labels) 
    all_preds_probs = torch.cat(all_preds_probs) 

    
    metrics_result = calculate_metrics(all_labels, all_preds_probs)
    
    return metrics_result


# In[10]:


def Student_Model():
    model = models_vit.load_pretrained_vit_base(target_size=Data_Image_size,global_pool=global_pool, num_classes=num_classes)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Student : - number of params (M): %.2f' % (n_parameters / 1.e6))

    param_groups = lrd.param_groups_lrd(model_without_ddp, 0.03,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=0.5
    )
    optimizer = torch.optim.AdamW(param_groups,weight_decay=0.02, lr=0.001)
    loss_scaler = NativeScaler()
    load_model(False,False, model_without_ddp=model_without_ddp, optimizer=optimizer,loss_scaler = loss_scaler)
    return model, optimizer, loss_scaler

def setup_logger():
    logger = logging.getLogger('my_logger')  
    logger.setLevel(logging.DEBUG) 

    file_handler = logging.FileHandler(output_directory  + 'Student_log.txt')
    file_handler.setLevel(logging.DEBUG)  
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    
    return logger

# In[12]:


if __name__ == '__main__':
    Data_Image_size = 200
    output_directory = './Student_Model/'
    torch.backends.cudnn.benchmark = True 
    torch.cuda.memory_summary(device=None, abbreviated=False)  
    Teacher_model_train, teacher_optimizer, Teacher_loss_scaler = RetFound_TeacherModel()
    Student_Model_train, student_optimizer,Student_loss_scaler = Student_Model()
    loss_scaler = NativeScaler()
    writer = SummaryWriter(log_dir=output_directory + "Student_tensorboard")
    logger = setup_logger()

    logger.info("Starting training...")
    root_dirs1 = './Glaucoma/'
    root_dirs2 = './DR/'
    root_dirs = [root_dirs1, root_dirs2]
    train_loader, val_loader, test_loader = build_combined_dataloaders(root_dirs=root_dirs, image_size = 200)
    accuracy_df = train_teacher_student(logger, Teacher_model_train, Student_Model_train,teacher_optimizer,student_optimizer, Teacher_loss_scaler,
                                        Student_loss_scaler,train_loader,val_loader=val_loader, test_loader=test_loader,epochs=100)
    accuracy_df.to_excel(output_directory + 'Student_metrics.xlsx', index=False)

