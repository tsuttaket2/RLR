from torch import FloatTensor, LongTensor, cat, mm, norm, randn, zeros, ones
from torch.nn import Parameter
from torch.nn import Module
from torch.autograd import Variable
from torch.nn.functional import sigmoid, log_softmax
import torch
from torch.nn import Linear, Module, ModuleList
from torch.nn.functional import relu

device = torch.device("cuda" if torch.cuda.is_available() == True else 'cpu')

class LSTM_MLP(torch.nn.Module):
    def __init__(self,input_dim=76,lstm_hidden_dim=128,mlp_hidden_dim=128,num_mlp_layers=1,num_classes=1,dropout=0.4):
        super(LSTM_MLP,self).__init__()
        self.LSTM=LSTM_model(input_dim=input_dim,hidden_dim=lstm_hidden_dim,dropout=dropout)
        self.MLP=MLP(input_dim=lstm_hidden_dim,hidden_layer_dim=mlp_hidden_dim,
                    num_layers=num_mlp_layers,num_classes=num_classes)
    def forward(self,input,mask):
        #Expect input Batch_Size x Input_Dim x Seq_Len
        h_t=self.LSTM(input,mask)
        
        (Batch_Size,hidden_dim)=(h_t.shape[0],h_t.shape[1])
        
        out = self.MLP(h_t).squeeze(-1)
        return out

class MLP(torch.nn.Module):
    """
    A multilayer perceptron with one hidden ReLU layer.
    Expects an input tensor of size (batch_size, input_dim) and returns
    a tensor of size (batch_size, output_dim).
    """
    def __init__(self,
                 input_dim,
                 hidden_layer_dim,
                 num_layers,
                 num_classes):
        super(MLP, self).__init__()

        self.num_layers = num_layers

        # create a list of layers of size num_layers
        layers = []
        for i in range(num_layers):
            d1 = input_dim if i == 0 else hidden_layer_dim
            d2 = hidden_layer_dim if i < (num_layers - 1) else num_classes
            layer = Linear(d1, d2)
            layers.append(layer)

        self.layers = ModuleList(layers)

    def forward(self, x):
        res = self.layers[0](x)
        for i in range(1, len(self.layers)):
            res = self.layers[i](relu(res))
        res=torch.sigmoid(res)
        return res

class LSTM_model(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,dropout=0.4):
        super(LSTM_model,self).__init__()
        self.hidden_dim=hidden_dim
        self.lstm = torch.nn.LSTM(input_size=input_dim,hidden_size=hidden_dim,
                                  num_layers=1, batch_first =True,dropout = dropout).to(device)
    
    def forward(self,input,mask):
        #Expect input Batch_Size x Input_Dim x Seq_Len
        #Change to Batch_Size x Seq_Len x Input_Dim 
        input=input.transpose(1,2)
        doc_lens=torch.sum(mask,dim=1)
        
        packed_seq_batch = torch.nn.utils.rnn.pack_padded_sequence(input, lengths=doc_lens, batch_first=True,enforce_sorted=False)
        #output, (hn, cn)
        _, (hn, _) = self.lstm(packed_seq_batch)
        
        return hn.squeeze(0)