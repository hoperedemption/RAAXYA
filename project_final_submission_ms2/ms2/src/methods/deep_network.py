from typing import List
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, Dataset, DataLoader
import numpy as np
from src.utils import accuracy_fn, macrof1_fn
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torchvision.transforms.v2 as transforms


## MS2

class CustomTensorDataset(Dataset):
    def __init__(self, images, gt_labels, transform=None, is_mlp=False):
        super().__init__()

        self.images = images
        self.gt_labels = gt_labels 
        self.transform = transform 

        self.is_mlp = is_mlp 
    
    def __getitem__(self, index):
        image = self.images[index]
        if self.is_mlp:
            image = torch.reshape(image, (28, 28))
        gt_label = self.gt_labels[index]
        
        if self.transform is not None:
            image = self.transform(image)
        
        if self.is_mlp:
            image = torch.reshape(image, (28 * 28,))

        return image, gt_label
    
    def __len__(self):
        return self.images.shape[0]

class CustomTransform(object):
    def __init__(self, params, transform, p=0.5):
        self.transform = transform(*params)
        self.p = p
    def __call__(self, image):
        if np.random.uniform(low=0.0, high=1.0) < self.p:
            return self.transform(image)
        else:
            return image

class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes, dimensions_=[512, 512], activations=['leaky relu', 'relu', 'tanh'], use_pca=False):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_size, n_classes, my_arg=32)
        
        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
        """
        super().__init__()

        self.n_classes = n_classes
        self.input_size = input_size
        self.dropout = nn.Dropout(p=0.25)

        self.use_pca = use_pca

        dimensions = []
        dimensions.append(input_size)
        dimensions += dimensions_
        dimensions.append(n_classes)
        
        self.activations = []
        self.n_layers = len(dimensions)

        tmp = [nn.Linear(dimensions[l-1], dimensions[l]) for l in range(1, self.n_layers)]
        self.linear_functions = nn.ModuleList(tmp)
        for fct in activations:
            if fct == "relu" : 
                self.activations.append(F.relu)
            elif fct == "sigmoid":
                self.activations.append(F.sigmoid)
            elif fct == "tanh":
                self.activations.append(lambda x : 1.71 * F.tanh(2/3 * x))
            elif fct == "elu":
                self.activations.append(F.elu)
            elif fct == "leaky relu":
                self.activations.append(F.leaky_relu)
            else:
                self.activations.append(F.relu)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        preds = x
        
        for i in range(self.n_layers-1):
            preds = self.activations[i](self.dropout(self.linear_functions[i](preds)))

        return preds #no softmax done


class ConvolutionLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, activation, groups=1, stride=(1, 1), bias=False):
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, groups=groups, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm2d(output_channels, affine=True)
        self.activ = activation
        
        layers: List[nn.Module] = [self.conv, self.batch_norm, self.activ]

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
    
class InvertedResidual(nn.Module):
    def __init__(self, input_channels, expantion_channels, output_channels, kernel_size, stride, use_hardswish):
        super().__init__()
        self.use_residual = stride == (1, 1) and input_channels == output_channels
        layers: List[nn.Module] = []
    
        self.activation = nn.SiLU() if use_hardswish == True else nn.ReLU()

        # expand convolution
        if input_channels != expantion_channels:
            layers.append(ConvolutionLayer(input_channels=input_channels, output_channels=expantion_channels, 
                                           kernel_size=(1, 1), activation=self.activation))
            
        # depthwise convolution
        layers.append(ConvolutionLayer(input_channels=expantion_channels, output_channels=expantion_channels, 
                                       groups=expantion_channels, stride=stride, kernel_size=kernel_size, activation=self.activation))
        
        # projection(pointwise) convolution
        layers.append(ConvolutionLayer(input_channels=expantion_channels, output_channels=output_channels,
                                       kernel_size=(1, 1), activation=nn.Identity()))
        
        # transform layers to block
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        r = self.block(x)
        if self.use_residual == True: 
            r = r + x
        return r
    
class InceptionModule(nn.Module):
    def __init__(self, input_channels, output_channels_one, expantion_channels_two, 
                  output_channels_two, expantion_channels_three, output_channels_three, 
                  pool_projection):
        super().__init__()

        block1: List[nn.Module] = [ConvolutionLayer(input_channels=input_channels, 
                                       output_channels=output_channels_one, 
                                       kernel_size=(1, 1), activation=nn.SiLU())]
        
        block2: List[nn.Module] = [InvertedResidual(input_channels=input_channels, 
                                                    expantion_channels=expantion_channels_two, 
                                                    output_channels=output_channels_two, 
                                                    kernel_size=(3, 3), 
                                                    stride=(1, 1), 
                                                    use_hardswish=True)]
        
        block3: List[nn.Module] = [InvertedResidual(input_channels=input_channels, 
                                                   expantion_channels=expantion_channels_three, 
                                                   output_channels=output_channels_three, 
                                                   kernel_size=(5, 5), 
                                                   stride=(1, 1), 
                                                   use_hardswish=True)]

        block4: List[nn.Module] = [
            nn.MaxPool2d((3, 3), stride=(1, 1), padding=(1, 1)), 
            ConvolutionLayer(input_channels=input_channels, output_channels=pool_projection, 
                             kernel_size=(1, 1), activation=nn.SiLU())
        ]

        all_blocks: List[List[nn.Module]] = [block1, block2, block3, block4]
        for block in all_blocks:
            block.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.block1 = nn.Sequential(*block1)
        self.block2 = nn.Sequential(*block2)
        self.block3 = nn.Sequential(*block3)
        self.block4 = nn.Sequential(*block4)
    
    def forward(self, x):
        y1 = self.block1(x)
        y2 = self.block2(x)
        y3 = self.block3(x)
        y4 = self.block4(x)

        return torch.cat([y1, y2, y3, y4], 1)
 


class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes, arch: Tuple[List[nn.Module], List[nn.Module]] = None):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)
        
        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        if arch is None: 
            conv_layers: List[nn.Module] = [ # 7 # 
            InvertedResidual(input_channels=input_channels, expantion_channels=88, output_channels=24, 
                             kernel_size=(3, 3), stride=(1, 1), use_hardswish=False),
            InvertedResidual(input_channels=24, expantion_channels=96, output_channels=40, 
                             kernel_size=(5, 5), stride=(2, 2), use_hardswish=True), 
            nn.Dropout(p=0.35),
            InvertedResidual(input_channels=40, expantion_channels=240, output_channels=40, 
                             kernel_size=(5, 5), stride=(1, 1), use_hardswish=True), 
            InvertedResidual(input_channels=40, expantion_channels=240, output_channels=40, 
                             kernel_size=(5, 5), stride=(1, 1), use_hardswish=True),
            InvertedResidual(input_channels=40, expantion_channels=120, output_channels=48, 
                             kernel_size=(5, 5), stride=(1, 1), use_hardswish=True),
            InvertedResidual(input_channels=48, expantion_channels=144, output_channels=48, 
                             kernel_size=(5, 5), stride=(1, 1), use_hardswish=True),
            InvertedResidual(input_channels=48, expantion_channels=288, output_channels=96, 
                             kernel_size=(5, 5), stride=(2, 2), use_hardswish=True),
            nn.Dropout(p=0.55),
            InvertedResidual(input_channels=96, expantion_channels=576, output_channels=96, 
                             kernel_size=(5, 5), stride=(1, 1), use_hardswish=True),
            InvertedResidual(input_channels=96, expantion_channels=576, output_channels=96, 
                             kernel_size=(5, 5), stride=(1, 1), use_hardswish=True),
            InvertedResidual(input_channels=96, expantion_channels=96, output_channels=576, 
                             kernel_size=(1, 1), stride=(1, 1), use_hardswish=True),
            nn.Dropout(p=0.6),
            nn.AdaptiveAvgPool2d((1, 1)), 
            InvertedResidual(input_channels=576, expantion_channels=576, output_channels=1024, 
                             kernel_size=(1, 1), stride=(1, 1), use_hardswish=True)
            ]

            fc: List[nn.Module] = [
                nn.Linear(1024, n_classes)
            ]
        else:
            conv_layers = arch[0]
            fc = arch[1]

        self.conv = nn.Sequential(*conv_layers)
        self.mlp = nn.Sequential(*fc)

        

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##

        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        return self.mlp(x)


class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"
        d_head = int(d / n_heads)
        self.d_head = d_head

        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):

                # Select the mapping associated to the given head.
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]

                # Map seq to q, k, v. 
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq) 

                attention = self.softmax(q @ k.T / (np.sqrt(self.d_head)))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)### WRITE YOUR CODE HERE using LayerNorm pytorch
        self.mhsa = MyMSA(hidden_d, n_heads) ### WRITE YOUR CODE HERE
        self.norm2 = nn.LayerNorm(hidden_d)### WRITE YOUR CODE HERE using LayerNorm pytorch
        self.mlp = nn.Sequential( ### WRITE YOUR CODE HERE
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Dropout(p=0.15),
            nn.Linear(mlp_ratio * hidden_d, hidden_d), 
            nn.Dropout(p=0.15)
        )

    def forward(self, x):
        # Write code for MHSA + residual connection.
        out =  self.norm2(x + self.mhsa(self.norm1(x)))
        # Write code for MLP(Norm(out)) + residual connection
        out = out + self.mlp(out)
        return out


class MyViT(nn.Module):

    def get_positional_embeddings(self, sequence_length, d):
        length = torch.arange(sequence_length)[:, None]
        dimensions = torch.arange(d)[None, :]
        
        angles = 1 / (torch.pow(10000, 2 * (dimensions // 2) / d))

        new_angles = length * angles

        pos_embeddings = torch.zeros_like(new_angles)

        pos_embeddings[:, ::2] = torch.sin(new_angles[:, ::2])
        pos_embeddings[:, 1::2] = torch.cos(new_angles[:, 1::2])

        if torch.cuda.is_available():
            device = torch.device("cuda")
            pos_embeddings = pos_embeddings.to(device)
        else:
            device = torch.device("cpu")
                
        return pos_embeddings

    def patchify(self, images, n_patches):
        n, c, h, w = images.shape
        assert h == w # We assume square image.
        patches = images.unfold(2, h // n_patches, h // n_patches).unfold(3, w // n_patches, w // n_patches)
        patches = patches.reshape(n, n_patches ** 2, h * w * c // n_patches ** 2)

        if torch.cuda.is_available():
            device = torch.device("cuda")
            patches = patches.to(device)
        else:
            device = torch.device("cpu")

        return patches
    
    """
    A Transformer-based neural network
    """
    
    def __init__(self, chw, n_patches, n_blocks, hidden_d, n_heads, out_d):
        """
        Initialize the network.
        
        """
        super().__init__()

        self.chw = chw # (C, H, W)
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d

        # Input and patches sizes
        assert chw[1] % n_patches == 0 # Input shape must be divisible by number of patches
        assert chw[2] % n_patches == 0
        self.patch_size =  (chw[1] / n_patches, chw[2] / n_patches)

        # Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # Positional embedding
        self.positional_embeddings = self.get_positional_embeddings(n_patches ** 2 + 1, hidden_d)

        # Transformer blocks
        self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

        # Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d)
        )

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        #import time


        n, c, h, w = x.shape

        # Divide images into patches.
        patches = self.patchify(x, self.n_patches) ### WRITE YOUR CODE HERE

        # Map the vector corresponding to each patch to the hidden size dimension.
        tokens = self.linear_mapper(patches)

        # Add classification token to the tokens.
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        
        # Add positional embedding.
        preds = tokens + self.positional_embeddings.repeat(n, 1, 1)

        
        for block in self.blocks:
            preds = block(preds)
        preds = preds[:, 0]
        preds = self.mlp(preds)

        return preds


class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size, optimizer_name="Adam", validation_set=None, validation_labels=None, validation=False, i=None):
        """
        Initialize the trainer object for a given model.

        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training
            batch_size (int): number of data points in each batch
        """
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.batch_size = batch_size
        self.i = i

        self.validation_set = validation_set
        self.validation_labels = validation_labels
        self.validation = validation

        self.criterion = nn.CrossEntropyLoss()
        if optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)  ### WRITE YOUR CODE HERE
        elif optimizer_name == "SGD":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr) # I am not sure 
        

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs. 
        
        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        epoch_avg_loss = np.zeros(self.epochs)
        epoch_training_accuracy = np.zeros(self.epochs)
        epoch_training_f1 = np.zeros(self.epochs)
        if self.validation == True:
            epoch_testing_accuracy = np.zeros(self.epochs)
            epoch_testing_f1 = np.zeros(self.epochs)
            max_f1_score = 0
            max_accuracy_score = 0 
        for ep in range(self.epochs):
            ret = self.train_one_epoch(dataloader, ep)

            ### WRITE YOUR CODE HERE if you want to do add something else at each epoch
            epoch_avg_loss[ep] = ret[0]
            epoch_training_accuracy[ep] = ret[1]
            epoch_training_f1[ep] = ret[2]
            if self.validation == True: 
                epoch_testing_accuracy[ep] = ret[3]
                epoch_testing_f1[ep] = ret[4]
                if max_accuracy_score < epoch_testing_accuracy[ep] and max_f1_score < epoch_testing_f1[ep]:
                    print("#### ---------------- Saved model --------------------------- ####")
                    max_accuracy_score = epoch_testing_accuracy[ep]
                    max_f1_score = epoch_testing_f1[ep]

                    
                    PATH = "best_model.pth.tar"
                    torch.save(self.model.state_dict(), PATH)

        epoch_range = np.arange(0, self.epochs)
        
        plt.figure(figsize=(20,20))
        plt.plot(epoch_range, epoch_training_accuracy)
        if self.validation == True:
            plt.plot(epoch_range, epoch_testing_accuracy)
        plt.ylim([0, 100])
        plt.xticks(epoch_range)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        if self.validation == True:
            plt.legend(['Training', 'Validation'], loc='upper left')
        else:
            plt.legend(['Training'], loc='upper left')
        plt.cla()

        plt.figure(figsize=(20,20))
        plt.plot(epoch_range, epoch_avg_loss)
        plt.xticks(epoch_range)
        plt.title('model avg training loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['Training loss'], loc='upper left')
        plt.cla()

        plt.figure(figsize=(20,20))
        plt.plot(epoch_range, epoch_training_f1)
        if self.validation == True:
            plt.plot(epoch_range, epoch_testing_f1)
        plt.xticks(epoch_range)
        plt.ylim([0, 1])
        plt.title('model f1 score')
        plt.ylabel('f1 score')
        plt.xlabel('epoch')
        if self.validation == True:
            plt.legend(['Training', 'Validation'], loc='upper left')
        else:
            plt.legend(['Training'], loc='upper left')
        plt.cla()




    def train_one_epoch(self, dataloader, ep):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##

        self.model.train()

        print(f"Epoch #({ep + 1})")
        print("-------------------------------------------------")

        avg_loss_over_epoch = 0
        index = 0

        predictions_epoch = np.zeros(len(dataloader.dataset))
        labels_epoch = np.zeros(len(dataloader.dataset))

        for i, data in enumerate(dataloader):
            inputs, labels = data

            if torch.cuda.is_available():
                device = torch.device("cuda")
                inputs, labels = inputs.to(device), labels.to(device)

            # zero out the gradient

            self.optimizer.zero_grad()
            
            predictions = self.model(inputs) 

            loss = self.criterion(predictions, labels)
            loss.backward()


            self.optimizer.step()

            avg_loss_over_epoch += loss.item()

            predictions_epoch[i * self.batch_size : min((i + 1) * self.batch_size, predictions_epoch.shape[0])] = np.argmax(predictions.detach().cpu().numpy(), axis=1)
            labels_epoch[i * self.batch_size: min((i + 1) * self.batch_size, labels_epoch.shape[0])] =  labels.detach().cpu().numpy()

            if i % self.batch_size == self.batch_size - 1:
                avg_loss_over_epoch /= self.batch_size 
                index = i

        
        training_accuracy_over_epoch = accuracy_fn(predictions_epoch, labels_epoch)
        training_f1_score_over_epoch = macrof1_fn(predictions_epoch, labels_epoch)

        print(f"Epoch #({ep + 1}), batch #{index + 1}, Loss : {avg_loss_over_epoch/self.batch_size}")
        print(f"Epoch #({ep + 1}), batch #{index + 1}, Training accuracy: {training_accuracy_over_epoch}")
        print(f"Epoch #({ep + 1}), batch #{index + 1}, Training F1 score: {training_f1_score_over_epoch}")

        if self.validation == True:
            validation_predictions = self.predict(self.validation_set).astype('int64')
            testing_accuracy_over_epoch = accuracy_fn(validation_predictions, self.validation_labels)
            testing_f1_over_epoch = macrof1_fn(validation_predictions, self.validation_labels)

            pred_count = np.bincount(validation_predictions, weights=validation_predictions==self.validation_labels, minlength=10)
            gt_count = np.bincount(self.validation_labels, minlength=10)
            gt_count_dp = np.where(gt_count > 0, gt_count, -1 * np.ones(10)) # doesn't make sense to count accuracy if label not even in gt
            accuracy_for_each_label = pred_count / gt_count_dp

            for labels in range(accuracy_for_each_label.shape[0]):
                print(f"Epoch #{ep + 1}, batch #{index + 1}, Accuracy for label #{labels} is: {accuracy_for_each_label[labels]}")

            print(f"Epoch #({ep + 1}), batch #{index + 1}, Testing accuracy: {testing_accuracy_over_epoch}")
            print(f"Epoch #({ep + 1}), batch #{index + 1}, Testing F1 score: {testing_f1_over_epoch}")
            to_return = (avg_loss_over_epoch, training_accuracy_over_epoch, training_f1_score_over_epoch, testing_accuracy_over_epoch, testing_f1_over_epoch)
        else:
            to_return = (avg_loss_over_epoch, training_accuracy_over_epoch, training_f1_score_over_epoch)

        print("-------------------------------------------------")

        return to_return



    def predict_torch(self, dataloader):
        """
        Predict the validation/test dataloader labels using the model.

        Hints:
            1. Don't forget to set your model to eval mode, i.e., self.model.eval()!
            2. You can use torch.no_grad() to turn off gradient computation, 
            which can save memory and speed up computation. Simply write:
                with torch.no_grad():
                    # Write your code here.

        Arguments:
            dataloader (DataLoader): dataloader for validation/test data
        Returns:
            pred_labels (torch.tensor): predicted labels of shape (N,),
                with N the number of data points in the validation/test data.
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        self.model.eval()

        predictions_overall = torch.zeros(len(dataloader.dataset))

        if torch.cuda.is_available():
            device = torch.device("cuda")
            predictions_overall = predictions_overall.to(device)
        else:
            device = torch.device("cpu")
            
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                inputs = data[0]

                if torch.cuda.is_available():
                    inputs = inputs.to(device)

                outputs = self.model(inputs)
                _, predictions = outputs.max(1)

                if torch.cuda.is_available():
                    predictions = predictions.to(device)

                predictions_overall[i * self.batch_size : min((i + 1) * self.batch_size, predictions_overall.shape[0])] = predictions

        return predictions_overall
        

    
    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """

        # First, prepare data for pytorch

        is_mlp = False
        if isinstance(self.model, MLP):
            my_tranform = transforms.Compose(
                [
                    #transforms.Resize((28, 28)),
                    transforms.ToImage(), 
                    transforms.RandomHorizontalFlip(p=0.5),
                    CustomTransform([(3, 3)], transforms.GaussianBlur, p=0.5), 
                    CustomTransform([(-10, 10), (0.2, 0.2)], transforms.RandomAffine, p=0.5),
                    CustomTransform([28, (0.7, 0.9)], transforms.RandomResizedCrop, p=0.5),
                    transforms.ToDtype(torch.float32, scale=True)
                    #transforms.Resize((28 * 28, 1))
                ]
            )

            is_mlp = True
        else:
            my_tranform = transforms.Compose(
                [
                    transforms.ToImage(), 
                    transforms.RandomHorizontalFlip(p=0.5),
                    CustomTransform([(3, 3)], transforms.GaussianBlur, p=0.5), 
                    CustomTransform([(-10, 10), (0.2, 0.2)], transforms.RandomAffine, p=0.5),
                    CustomTransform([28, (0.7, 0.9)], transforms.RandomResizedCrop, p=0.5),
                    transforms.ToDtype(torch.float32, scale=True)
                ]
            )
        
        train_dataset = CustomTensorDataset(torch.from_numpy(training_data).float(), 
                                       torch.from_numpy(training_labels), transform=my_tranform, is_mlp=is_mlp)
        
        if isinstance(self.model, MLP) and self.model.use_pca == True:
            train_dataset = TensorDataset(torch.from_numpy(training_data).float(), 
                                       torch.from_numpy(training_labels))
            
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        self.train_all(train_dataloader)

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        This serves as an interface between numpy and pytorch.
        
        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # First, prepare data for pytorch
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        pred_labels = self.predict_torch(test_dataloader)

        # We return the labels after transforming them into numpy array.
        return pred_labels.cpu().numpy()