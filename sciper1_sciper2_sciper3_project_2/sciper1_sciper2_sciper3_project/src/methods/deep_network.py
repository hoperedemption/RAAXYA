from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from src.utils import accuracy_fn, macrof1_fn

## MS2


class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes, dimensions_=[512, 512], activations=["relu", "relu", "tanh"]):
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


        dimensions = []
        dimensions.append(input_size)
        dimensions += dimensions_
        dimensions.append(n_classes)
        
        self.activations = []
        self.n_layers = len(dimensions)

        # for l in range(1, self.n_layers):
        #     #self.w[l] = torch.normal(torch.zeros(dimensions[l-1], dimensions[l]), torch.ones(dimensions[l-1], dimensions[l]))
        #     self.w[l] = nn.Linear(dimensions[l-1], dimensions[l])
        #     #self.b[l] = torch.zeros(dimensions[l])
        #     self.activations[l] = activations[l-1]

        tmp = [nn.Linear(dimensions[l-1], dimensions[l]) for l in range(1, self.n_layers)]
        self.linear_functions = nn.ModuleList(tmp)
        for fct in activations:
            if fct == "relu" : 
                self.activations.append(F.relu)
            elif fct == "sigmoid":
                self.activations.append(F.sigmoid)
            elif fct == "tanh":
                self.activations.append(lambda x : 1.71 * F.tanh(2/3 * x))

        # self.loss = nn.CrossEntropyLoss(reduction='none')
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
            #preds = self.activations[i](F.dropout(self.linear_functions[i](preds), p=0.5))
            preds = self.activations[i](self.linear_functions[i](preds))

        return preds #no softmax done


class ConvolutionLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, groups=1, stride=(1, 1), bias=False, activation=F.relu):
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, groups=groups, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm2d(output_channels, affine=True)
        self.layer = nn.Sequential(
            self.conv, 
            self.batch_norm
        )

        self.activ = activation

    def forward(self, x):
        return self.activ(self.layer(x))
    
class InvertedResidual(nn.Module):
    def __init__(self, input_channels, expantion_channels, output_channels, kernel_size, stride, use_hardswish):
        super().__init__()
        self.use_residual = stride == (1, 1) and input_channels == output_channels
        layers: List[nn.Module] = []
    
        self.activation = F.hardswish if use_hardswish == True else F.relu

        # expand convolution
        if input_channels != expantion_channels:
            layers.append(ConvolutionLayer(input_channels=input_channels, output_channels=expantion_channels, 
                                           kernel_size=(1, 1), activation=self.activation))
            
        # depthwise convolution
        layers.append(ConvolutionLayer(input_channels=expantion_channels, output_channels=expantion_channels, 
                                       groups=expantion_channels, stride=stride, kernel_size=kernel_size, activation=self.activation))
        
        # projection(pointwise) convolution
        layers.append(ConvolutionLayer(input_channels=expantion_channels, output_channels=output_channels,
                                       kernel_size=(1, 1), activation=F.relu))
        
        # transform layers to block
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        r = self.block(x)
        if self.use_residual == True: 
            r = r + x
        return r



class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes):
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

        self.arch: List[nn.Module] = [
            InvertedResidual(input_channels=input_channels, expantion_channels=64, output_channels=24, 
                            kernel_size=(3, 3), stride=(1, 1), use_hardswish=False),
            InvertedResidual(input_channels=24, expantion_channels=96, output_channels=40,
                            kernel_size=(3, 3), stride=(1, 1), use_hardswish=False),
            InvertedResidual(input_channels=40, expantion_channels=240, output_channels=40,
                            kernel_size=(5, 5), stride=(1, 1), use_hardswish=True), 
            InvertedResidual(input_channels=40, expantion_channels=240, output_channels=40,
                            kernel_size=(5, 5), stride=(1, 1), use_hardswish=True), 
            InvertedResidual(input_channels=40, expantion_channels=120, output_channels=48,
                            kernel_size=(5, 5), stride=(1, 1), use_hardswish=True), 
            InvertedResidual(input_channels=48, expantion_channels=144, output_channels=48, 
                            kernel_size=(5, 5), stride=(1, 1), use_hardswish=True),
            InvertedResidual(input_channels=48, expantion_channels=288, output_channels=96, 
                            kernel_size=(5, 5), stride=(1, 1), use_hardswish=True),
            InvertedResidual(input_channels=96, expantion_channels=576, output_channels=96, 
                            kernel_size=(5, 5), stride=(1, 1), use_hardswish=True),
            InvertedResidual(input_channels=96, expantion_channels=576, output_channels=96, 
                            kernel_size=(5, 5), stride=(1, 1), use_hardswish=True),
            InvertedResidual(input_channels=96, expantion_channels=96, output_channels=576, 
                            kernel_size=(5, 5), stride=(1, 1), use_hardswish=True)
        ]
        
        self.block = nn.Sequential(*self.arch)
        
        self.fc_1 = nn.Linear(576, 128)
        self.fc_2 = nn.Linear(128, n_classes)

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

        # ------------------------- convolution --------------------
        for i, layer in enumerate(self.arch):
            if i == 2: # first reduction
                x = F.adaptive_avg_pool2d(x, (14, 14))
            elif i == 7: # second reduction 
                x = F.adaptive_avg_pool2d(x, (7, 7))
            x = layer(x)
        # ------------------------- convolution --------------------

        x = F.adaptive_avg_pool2d(x, (1, 1))
        # no normalisation

        # fully connected part
        x = x.reshape(x.shape[0], -1)
        x = F.relu(F.dropout(self.fc_1(x), p=0.5))
        return self.fc_2(x)


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
                q, k, v = q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq) ### WRITE YOUR CODE HERE

                attention = self.softmax(q @ k.T / (np.sqrt(self.d_head))) ### WRITE YOUR CODE HERE
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
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        # Write code for MHSA + residual connection.
        out =  self.norm2(x + self.mhsa(self.norm1(x)))
        # Write code for MLP(Norm(out)) + residual connection
        out = out + self.mlp(out)
        return out


class MyViT(nn.Module):

    def get_positional_embeddings(self, sequence_length, d):
        result = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                if j % 2 == 0:
                    result[i][j] = np.sin(i / (10000 ** (j / d)))
                else:
                    result[i][j] = np.cos(i / (10000 ** ((j - 1) / d)))
                
        return result

    def patchify(self, images, n_patches):
        n, c, h, w = images.shape
        assert h == w # We assume square image.
        patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
        patch_size = h // n_patches

        for idx, image in enumerate(images):
            for i in range(n_patches):
                for j in range(n_patches):
                    patch = image[:,i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                    patches[idx, i * n_patches + j] = patch.flatten()
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
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
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

        n, c, h, w = x.shape

        # Divide images into patches.
        patches = self.patchify(x, self.n_patches) ### WRITE YOUR CODE HERE

        # Map the vector corresponding to each patch to the hidden size dimension.
        tokens = self.linear_mapper(patches)

        # Add classification token to the tokens.
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Add positional embedding.
        # HINT: use torch.Tensor.repeat(...)
        preds = tokens + self.positional_embeddings.repeat(n, 1, 1)### WRITE YOUR CODE HERE

        # Transformer Blocks
        for block in self.blocks:
            preds = block(preds)

        # Get the classification token only.
        preds = preds[:, 0]

        # Map to the output distribution.
        preds = self.mlp(preds)

        return preds


class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size, optimizer_name="Adam", validation_set=None, validation_labels=None, validation=False):
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
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader, ep)

            ### WRITE YOUR CODE HERE if you want to do add something else at each epoch

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

        predictions_epoch = np.array([])
        labels_epoch = np.array([])

        for i, data in enumerate(dataloader):
            inputs, labels = data

            # zero out the gradient
            self.optimizer.zero_grad()

            predictions = self.model(inputs)

            loss = self.criterion(predictions, labels)
            loss.backward()

            self.optimizer.step()

            avg_loss_over_epoch += loss.item()

            predictions_epoch = np.append(predictions_epoch, np.argmax(predictions.detach().cpu().numpy(), axis=1))
            labels_epoch = np.append(labels_epoch, labels.detach().cpu().numpy())

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

            accuracy_for_each_label = np.bincount(validation_predictions, weights=(validation_predictions==self.validation_labels)) / np.bincount(self.validation_labels)
            for labels in range(accuracy_for_each_label.shape[0]):
                print(f"Epoch #{ep + 1}, batch #{index + 1}, Accuracy for label #{labels} is: {accuracy_for_each_label[labels]}")

            print(f"Epoch #({ep + 1}), batch #{index + 1}, Testing accuracy: {testing_accuracy_over_epoch}")
            print(f"Epoch #({ep + 1}), batch #{index + 1}, Testing F1 score: {testing_f1_over_epoch}")

        print("-------------------------------------------------")


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

        preds = torch.tensor([])
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                inputs = data[0]
                outputs = self.model(inputs)
                _, predictions = outputs.max(1)
                preds = torch.cat((preds, predictions), dim=-1)
        return preds
        

    
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