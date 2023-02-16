

from torch import FloatTensor, LongTensor, cat, mm, norm, randn, zeros, ones
from torch.nn import Parameter
from torch.nn import Module
from torch.autograd import Variable
from torch.nn.functional import sigmoid, log_softmax, tanh
import torch
from collections import OrderedDict

from .my_utils import enable_gradient_clipping, mask_sst, remove_old, to_file, shuffled_chunked_sorted, chunked_sorted, Batch
from torch.nn import NLLLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import log_softmax
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Combine SoPa and MLP
class SoPa_MLP(Module):
    def __init__(self,
                 input_dim,
                 pattern_specs,
                 semiring,
                 mlp_hidden_dim,
                 num_mlp_layers,
                 num_classes,
                 embeddings,
                 deep_supervision=False,
                 gpu=True,
                 dropout=0.4,
                 vocab=None):
        super(SoPa_MLP,self).__init__()

        self.deep_supervision=deep_supervision
        self.total_num_patterns = sum(pattern_specs.values())
        
        self.sopa = SoftPatternClassifier(input_dim,pattern_specs,semiring,deep_supervision,gpu,dropout)
        self.mlp = MLP(self.total_num_patterns, mlp_hidden_dim, num_mlp_layers, num_classes)

        self.to_cuda =self.sopa.to_cuda
        self.dropout=dropout
        self.embeddings = embeddings
        self.semiring = semiring
        self.num_classes = num_classes
        self.mlp_hidden_dim = mlp_hidden_dim
        self.num_mlp_layers = num_mlp_layers
        self.vocab = vocab
    def forward(self,input,mask):

        if self.deep_supervision :
            s_t=self.sopa(input,mask,self.dropout)
            out=self.to_cuda(torch.zeros((s_t.shape[0],s_t.shape[2])))
            for t in range(s_t.shape[2]):
                out[:,t]=self.mlp(s_t[...,t]).squeeze(-1)
            #Output will be Batch_Size x Prediction x Time Length
            return out
        else :
            s_doc = self.sopa(input,mask,self.dropout)
            return (self.mlp(s_doc)).squeeze(-1)

    def predict(self, input, mask):
        output = self.forward(input, mask)
        return [int(x) for x in argmax(output)]


def identity(x):
    return x
def neg_infinity(*sizes):
    return -100 * ones(*sizes)  # not really -inf, shh
def to_cuda(gpu):
    return (lambda v: v.cuda()) if gpu else identity
def fixed_var(tensor):
    return Variable(tensor, requires_grad=False)
def argmax(output):
    """ only works for kxn tensors """
    _, am = torch.max(output, 1)
    return am


class Semiring:
    def __init__(self,
                 zero,
                 one,
                 plus,
                 times,
                 from_float,
                 to_float):
        self.zero = zero
        self.one = one
        self.plus = plus
        self.times = times
        self.from_float = from_float
        self.to_float = to_float
MaxPlusSemiring = \
    Semiring(
        neg_infinity,
        zeros,
        torch.max,
        torch.add,
        identity,
        identity
    )
ProbSemiring = \
    Semiring(
        zeros,
        ones,
        torch.add,
        torch.mul,
        sigmoid,
        identity
    )
ProbSemiring2 = \
    Semiring(
        zeros,
        ones,
        torch.add,
        torch.mul,
        tanh,
        identity
    )
# element-wise max, times. in log-space
LogSpaceMaxTimesSemiring = \
    Semiring(
        neg_infinity,
        zeros,
        torch.max,
        torch.add,
        lambda x: torch.log(torch.min(torch.sigmoid(x)+1e-7,torch.tensor([1.]).to(device))),
        torch.exp
    )
def normalize(data):
    length = data.shape[0]
    for i in range(length):
        data[i] = data[i] / norm(data[i])  # unit length

def get_regularization_groups(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_weights_params = torch.ones_like(model.diags.view(model.total_num_patterns,-1,model.input_dim)).to(device)
    n_bias_params = torch.ones_like(model.bias.view(model.total_num_patterns,model.num_diags * model.max_pattern_length)).to(device)
    n_weights_params=n_weights_params.sum(dim=1).sum(dim=1)
    n_bias_params = n_bias_params.sum(dim=1)

    reshaped_weights = model.diags.view(model.total_num_patterns,-1,model.input_dim)
    reshaped_bias = model.bias.view(model.total_num_patterns,model.num_diags * model.max_pattern_length)
    l2_norm = reshaped_weights.norm(2, dim=1).norm(2, dim=1)/n_weights_params + reshaped_bias.norm(2, dim=1)/n_bias_params
    return l2_norm

THRESHOLD=0.1
def extract_learned_structure(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    regularization_groups = get_regularization_groups(model.sopa)
    end_states = [ [end] for pattern_len, num_patterns in model.sopa.pattern_specs.items()
                            for end in num_patterns * [pattern_len - 1] ]
    assert len(end_states) == len(regularization_groups)
    
    #Boolean tensor telling which pattern should be deleted
    WFSA_delete = regularization_groups < THRESHOLD
    new_num_states=[e[0]+1 for wfsa,e in zip(WFSA_delete,end_states) if torch.logical_not(wfsa) ]
    new_states=set(new_num_states)
    new_pattern_specs = OrderedDict.fromkeys(sorted(new_states),0)
    if new_pattern_specs:

        for i in new_num_states:
            new_pattern_specs[i]+=1
        Last_layer=model.mlp.layers[model.mlp.num_layers-1]
        
        new_model=SoPa_MLP(input_dim=300,
                            pattern_specs=new_pattern_specs,
                            semiring=model.sopa.semiring ,
                            mlp_hidden_dim=model.mlp_hidden_dim,
                            num_mlp_layers=model.num_mlp_layers,
                            num_classes=model.num_classes,
                            embeddings=model.embeddings,
                            deep_supervision=model.deep_supervision,
                            gpu=True,
                            dropout=model.dropout).to(device)
        
        #New model parameters
        new_weights=new_model.sopa.diags.reshape(new_model.sopa.total_num_patterns , new_model.sopa.num_diags , new_model.sopa.max_pattern_length,new_model.sopa.diags.shape[1])
        new_biases=new_model.sopa.bias.reshape(new_model.sopa.total_num_patterns , new_model.sopa.num_diags , new_model.sopa.max_pattern_length,1)
        new_epsilon=new_model.sopa.epsilon
        #Old model parameters
        old_weights= model.sopa.diags.reshape(model.sopa.total_num_patterns , model.sopa.num_diags , model.sopa.max_pattern_length,model.sopa.diags.shape[1])
        old_biases= model.sopa.bias.reshape(model.sopa.total_num_patterns , model.sopa.num_diags , model.sopa.max_pattern_length,1)
        old_epsilon=model.sopa.epsilon
        
        #Copy the params values from old to new
        WFSA_count=0
        for n, WFSA in enumerate(WFSA_delete):
            if not(WFSA.item()):
                new_weights[WFSA_count,:,:,:] = old_weights[n,:,:new_weights.shape[2],:]
                new_biases[WFSA_count,:,:,:] = old_biases[n,:,:new_biases.shape[2],:]
                new_epsilon[WFSA_count,:] = old_epsilon[n,:new_epsilon.shape[1]]
                WFSA_count+=1
        
        new_weights = new_weights.reshape(-1,new_weights.shape[-1])
        new_biases = new_biases.reshape(-1,new_biases.shape[-1])
        
        new_model.sopa.diags=Parameter(new_weights)
        new_model.sopa.bias=Parameter(new_biases)
        new_model.sopa.epsilon=Parameter(new_epsilon)
    
        return new_model, new_pattern_specs
    else:
        return None, None


class SoftPatternClassifier(Module):
    """
    A text classification model that feeds the document scores from a bunch of
    soft patterns into an MLP
    """
    def __init__(self,
                 input_dim,
                 pattern_specs,
                 semiring,
                 deep_supervision=False,
                 gpu=True,
                 dropout=0.4):
        super(SoftPatternClassifier, self).__init__()
        self.to_cuda = to_cuda(gpu)

        self.pattern_specs = pattern_specs
        self.max_pattern_length = max(list(pattern_specs.keys()))
        
        self.semiring=semiring
        self.input_dim=input_dim
        self.deep_supervision = deep_supervision
        # end state index for each pattern
        end_states = [
            [end]
            for pattern_len, num_patterns in self.pattern_specs.items()
            for end in num_patterns * [pattern_len - 1]
        ]

        self.end_states = self.to_cuda(fixed_var(LongTensor(end_states)))

        self.num_diags=2
        #out channel
        self.total_num_patterns = sum(pattern_specs.values())
        

        diag_data_size = self.total_num_patterns * self.num_diags * self.max_pattern_length
        diag_data = randn(diag_data_size, input_dim)
        bias_data = randn(diag_data_size, 1)
        
        normalize(diag_data)

        self.diags = Parameter(diag_data)

        # Bias term
        self.bias = Parameter(bias_data)

        self.epsilon = Parameter(randn(self.total_num_patterns, self.max_pattern_length - 1))

        self.epsilon_scale = self.to_cuda(fixed_var(semiring.one(1)))
        
        self.dropout = torch.nn.Dropout(dropout)
        print("# params:", sum(p.nelement() for p in self.parameters()))
        
    def get_transition_matrices(self, batch,dropout):
        b = batch.size()[0]
        n = batch.size()[2]
        assert self.input_dim==batch.size()[1]
        
        #transition_scores = \
        #        self.semiring.from_float(mm(self.diags, batch.embeddings_matrix) + self.bias_scale_param * self.bias).t()
            
        transition_scores = \
            self.semiring.from_float(torch.einsum('di,bin->bnd',self.diags,batch) + self.bias.squeeze(1) )
        
        if dropout is not None and dropout:
            transition_scores = self.dropout(transition_scores)
        
        
        batched_transition_scores = transition_scores.view(
            b, n, self.total_num_patterns, self.num_diags, self.max_pattern_length)
        return batched_transition_scores.transpose(0,1)
    def get_eps_value(self):
        return self.semiring.times(
            self.epsilon_scale,
            self.semiring.from_float(self.epsilon)
        )
    def forward(self, batch, mask, dropout=None):
        """ Calculate scores for one batch of documents. """
        transition_matrices = self.get_transition_matrices(batch,dropout)
        
        
        batch_size = batch.size()[0]
        num_patterns = self.total_num_patterns
        n = batch.size()[2]
        doc_lens=torch.sum(mask,dim=1)

        scores = self.to_cuda(fixed_var(self.semiring.zero((batch_size, num_patterns))))
        
        # to add start state for each word in the document.
        restart_padding = self.to_cuda(fixed_var(self.semiring.one(batch_size, num_patterns, 1)))

        zero_padding = self.to_cuda(fixed_var(self.semiring.zero(batch_size, num_patterns, 1)))

        eps_value = self.get_eps_value()

        batch_end_state_idxs = self.end_states.expand(batch_size, num_patterns, 1)
        hiddens = self.to_cuda(Variable(self.semiring.zero(batch_size,
                                                           num_patterns,
                                                           self.max_pattern_length)))
        if self.deep_supervision:
            s_t = self.to_cuda(self.semiring.zero(batch_size, num_patterns,n))
        
        # set start state (0) to 1 for each pattern in each doc
        hiddens[:, :, 0] = self.to_cuda(self.semiring.one(batch_size, num_patterns))
        
            
        for i, transition_matrix in enumerate(transition_matrices):
            self_loop_scale=None
            hiddens = self.transition_once(eps_value,
                                           hiddens,
                                           transition_matrix,
                                           zero_padding,
                                           restart_padding,
                                           self_loop_scale)
            
            # Look at the end state for each pattern, and "add" it into score
            end_state_vals = torch.gather(hiddens, 2, batch_end_state_idxs).view(batch_size, num_patterns)  #Equation8a
            if self.deep_supervision:
                s_t[...,i]=self.semiring.to_float(end_state_vals) #Equation8b
      
            # but only update score when we're not already past the end of the doc
            active_doc_idxs = torch.nonzero(torch.gt(doc_lens, i)).squeeze()

            scores[active_doc_idxs] = \
                self.semiring.plus(
                    scores[active_doc_idxs].clone(),
                    end_state_vals[active_doc_idxs]
                )
        
        scores = self.semiring.to_float(scores)
        if self.deep_supervision :
            return s_t
        else:
            return scores
    def transition_once(self,
                        eps_value,
                        hiddens,
                        transition_matrix_val,
                        zero_padding,
                        restart_padding,
                        self_loop_scale):
        after_epsilons = self.semiring.plus(hiddens,cat((zero_padding,self.semiring.times(hiddens[:, :, :-1],eps_value)), 2))
        
        after_main_paths = \
            cat((restart_padding,  # <- Adding the start state
                 self.semiring.times(
                     after_epsilons[:, :, :-1],
                     transition_matrix_val[:, :, -1, :-1])
                 ), 2)
        
        after_self_loops = \
                self.semiring.times(
                    after_epsilons,
                    transition_matrix_val[:, :, 0, :]
                )
        return self.semiring.plus(after_main_paths, after_self_loops)

from argparse import ArgumentParser

from torch.nn import Linear, Module, ModuleList
from torch.nn.functional import relu


class MLP(Module):
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
        return res

def evaluate_accuracy(model, data, batch_size, gpu):
    total_number = float(len(data))
    correct = 0
    num_1s = 0
    for batch in chunked_sorted(data, batch_size):
        
        batch_obj = Batch([x for x, y in batch], model.embeddings, to_cuda(gpu))
        gold = [y for x, y in batch]
        
        batch_x_temp = [torch.index_select(batch_obj.embeddings_matrix, 1, doc) for doc in batch_obj.docs]
        b = len(batch_x_temp)
        n = batch_obj.max_doc_len
        input_dim = batch_x_temp[0].shape[0]
        batch_x = torch.zeros((b,input_dim,n)).to(device)
        for i,doc_len in enumerate(batch_obj.doc_lens):
            batch_x[i,:,:] = batch_x_temp[i]
        batch_mask = torch.zeros((b,n)).to(device)
        batch_mask = mask_sst(batch_obj.doc_lens,batch_mask)

        predicted = model.predict(batch_x, batch_mask)
        num_1s += predicted.count(1)
        correct += sum(1 for pred, gold in zip(predicted, gold) if pred == gold)

    print("num predicted 1s:", num_1s)
    print("num gold 1s:     ", sum(gold == 1 for _, gold in data))
    
    return correct / total_number

def train_reg_str(train_data, dev_data, model, num_classes, num_iterations, learning_rate, batch_size,run_scheduler, gpu, clip, max_len, patience, reg_strength, logging_path):
    """ Train a model on all the given docs """
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_function = NLLLoss(None, False)
    
    enable_gradient_clipping(model, clip)

    if run_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, 'min', 0.1, 10, True)

    best_dev_loss = 100000000
    best_dev_loss_index = -1
    best_dev_acc = -1
    unchanged = 0
    reduced_model_path = ""
    learned_pattern_specs = ""
    stop = False
    for it in range(num_iterations):
        np.random.shuffle(train_data)

        loss = 0.0
        model.train()
        for batch in shuffled_chunked_sorted(train_data, batch_size):
            optimizer.zero_grad()
            batch_obj = Batch([x[0] for x in batch], model.embeddings, to_cuda(gpu), 0, max_len)
            batch_y = torch.tensor([x[1] for x in batch]).to(device)
            
            batch_x_temp = [torch.index_select(batch_obj.embeddings_matrix, 1, doc) for doc in batch_obj.docs]
            b = len(batch_x_temp)
            n = batch_obj.max_doc_len
            input_dim = batch_x_temp[0].shape[0]
            batch_x = torch.zeros((b,input_dim,n)).to(device)
            for i,doc_len in enumerate(batch_obj.doc_lens):
                batch_x[i,:,:] = batch_x_temp[i]
            batch_mask = torch.zeros((b,n)).to(device)
            batch_mask = mask_sst(batch_obj.doc_lens,batch_mask)
            batch_y_hat = model(batch_x,batch_mask)
            
            loss = loss_function(log_softmax(batch_y_hat).view(b, 2),batch_y)
            
            #Regularization 
            regularization_term= get_regularization_groups(model.sopa)
            regularization_term = torch.sum(regularization_term)
            reg_loss = reg_strength * regularization_term
            train_loss = loss + reg_loss
        
            
            train_loss.backward()
            optimizer.step()
            
        model.eval()
        train_acc = evaluate_accuracy(model, train_data[:1000], batch_size, True)
        dev_acc = evaluate_accuracy(model, dev_data, batch_size, True)
        print("train_acc ", train_acc)
        print("dev_acc ", dev_acc)
            
        if best_dev_acc < dev_acc:
            unchanged = 0
            best_dev_acc=dev_acc
        else:
            unchanged += 1
        if unchanged >= patience:
            stop = True

        epoch_string = "\n"
        epoch_string += "-" * 110 + "\n"
        epoch_string += "| Epoch={} | reg_strength={} | train_loss={:.6f} | valid_acc={:.6f} | regularized_loss={:.6f} |".format(
            it,
            reg_strength,
            train_loss.item(),
            dev_acc,
            reg_loss.item()
            )
        
        new_model_valid_err = -1.0
        new_model, new_pattern_specs = extract_learned_structure(model)
        if new_model is not None:
            if new_pattern_specs != model.sopa.pattern_specs:
                new_model_dev_acc = evaluate_accuracy(new_model, dev_data, batch_size, True)
            else:
                new_model_dev_acc = dev_acc
        epoch_string += " extracted_structure valid_err={:.6f} |".format(new_model_dev_acc)
        print(epoch_string)
        
        if unchanged == 0:
            if reduced_model_path != "":
                remove_old(reduced_model_path)
         
            learned_pattern_specs, reduced_model_path = to_file(new_model,new_pattern_specs,logging_path)
            


        if stop:
            break
    return dev_acc, learned_pattern_specs, reduced_model_path
    
def train_no_reg(train_data, dev_data, model, num_classes, num_iterations, learning_rate, batch_size,run_scheduler, gpu, clip, max_len, patience, logging_path):
    """ Train a model on all the given docs """
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_function = NLLLoss(None, False)
    
    enable_gradient_clipping(model, clip)

    if run_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, 'min', 0.1, 10, True)

    best_dev_loss = 100000000
    best_dev_loss_index = -1
    best_dev_acc = -1
    unchanged = 0
    reduced_model_path = ""
    stop = False
    for it in range(num_iterations):
        np.random.shuffle(train_data)

        loss = 0.0
        model.train()
        for batch in shuffled_chunked_sorted(train_data, batch_size):
            optimizer.zero_grad()
            batch_obj = Batch([x[0] for x in batch], model.embeddings, to_cuda(gpu), 0, max_len)
            batch_y = torch.tensor([x[1] for x in batch]).to(device)
            
            batch_x_temp = [torch.index_select(batch_obj.embeddings_matrix, 1, doc) for doc in batch_obj.docs]
            b = len(batch_x_temp)
            n = batch_obj.max_doc_len
            input_dim = batch_x_temp[0].shape[0]
            batch_x = torch.zeros((b,input_dim,n)).to(device)
            for i,doc_len in enumerate(batch_obj.doc_lens):
                batch_x[i,:,:] = batch_x_temp[i]
            batch_mask = torch.zeros((b,n)).to(device)
            batch_mask = mask_sst(batch_obj.doc_lens,batch_mask)
            batch_y_hat = model(batch_x,batch_mask)
            
            loss = loss_function(log_softmax(batch_y_hat).view(b, 2),batch_y)
            
            train_loss = loss
        
            
            train_loss.backward()
            optimizer.step()
            
        model.eval()
        train_acc = evaluate_accuracy(model, train_data[:1000], batch_size, True)
        dev_acc = evaluate_accuracy(model, dev_data, batch_size, True)
        print("train_acc ", train_acc)
        print("dev_acc ", dev_acc)
            
        if best_dev_acc < dev_acc:
            unchanged = 0
            best_dev_acc=dev_acc
        else:
            unchanged += 1
        if unchanged >= patience:
            stop = True

        epoch_string = "\n"
        epoch_string += "-" * 110 + "\n"
        epoch_string += "| Epoch={} | train_loss={:.6f} | valid_acc={:.6f} |".format(
            it,
            train_loss.item(),
            dev_acc
            )
        
        print(epoch_string)
        
        if unchanged == 0:
            if reduced_model_path != "":
                remove_old(reduced_model_path)
         
            pattern_specs, reduced_model_path = to_file(model,model.sopa.pattern_specs,logging_path)
            


        if stop:
            break
    return best_dev_acc, pattern_specs, reduced_model_path

from functools import total_ordering
@total_ordering
class BackPointer:
    def __init__(self,
                 score,
                 previous,
                 transition,
                 start_token_idx,
                 end_token_idx):
        self.score = score
        self.previous = previous
        self.transition = transition
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx

    def __eq__(self, other):
        return self.score == other.score

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        return self.score < other.score

    def __repr__(self):
        return \
            "BackPointer(" \
                "score={}, " \
                "previous={}, " \
                "transition={}, " \
                "start_token_idx={}, " \
                "end_token_idx={}" \
            ")".format(
                self.score,
                self.previous,
                self.transition,
                self.start_token_idx,
                self.end_token_idx
            )

    def display(self, doc_text, extra=""):
        if self.previous is None:
            #print(" ".join(doc_text))
            return extra  # " ".join("{:<15}".format(s) for s in doc_text[self.start_token_idx:self.end_token_idx])
        if self.transition == "self-loop":
            extra = "SL {:<15} \n".format(doc_text[self.end_token_idx - 1]) + extra
            return self.previous.display(doc_text, extra=extra)
        if self.transition == "happy path":
            extra = "HP {:<15} \n".format(doc_text[self.end_token_idx - 1]) + extra
            return self.previous.display(doc_text, extra=extra)
        extra = "ep {:<15} \n".format("") + extra
        return self.previous.display(doc_text, extra=extra)

def zip_ap_2d(f, a, b):
    return [
        [
            f(x, y) for x, y in zip(xs, ys)
        ]
        for xs, ys in zip(a, b)
    ]


def cat_2d(padding, a):
    return [
        [p] + xs
        for p, xs in zip(padding, a)
    ]



def transition_once_with_trace(model,
                               token_idx,
                               eps_value,
                               back_pointers,
                               transition_matrix_val,
                               restart_padding,
                               zero_padding):
    def times(a, b):
        # wildly inefficient, oh well
        return model.semiring.times(
            torch.FloatTensor([a]),
            torch.FloatTensor([b])
        )[0]

    # Epsilon transitions (don't consume a token, move forward one state)
    # We do this before self-loops and single-steps.
    # We only allow one epsilon transition in a row.
    epsilons = cat_2d(
        zero_padding(token_idx),
        zip_ap_2d(
            lambda bp, e: BackPointer(score=times(bp.score, e),
                                    previous=bp,
                                    transition="epsilon-transition",
                                    start_token_idx=bp.start_token_idx,
                                    end_token_idx=token_idx),
            [xs[:-1] for xs in back_pointers],
            eps_value  # doesn't depend on token, just state
        )
    )

    epsilons = zip_ap_2d(max, back_pointers, epsilons)
    
    happy_paths = cat_2d(
        restart_padding(token_idx),
        zip_ap_2d(
            lambda bp, t: BackPointer(score=times(bp.score, t),
                                    previous=bp,
                                    transition="happy path",
                                    start_token_idx=bp.start_token_idx,
                                    end_token_idx=token_idx + 1),
            [xs[:-1] for xs in epsilons],
            transition_matrix_val[:, 1, :-1]
        )
    )

    # Adding self loops (consume a token, stay in same state)
    self_loops = zip_ap_2d(
        lambda bp, sl: BackPointer(score=times(bp.score, sl),
                                previous=bp,
                                transition="self-loop",
                                start_token_idx=bp.start_token_idx,
                                end_token_idx=token_idx + 1),
        epsilons,
        transition_matrix_val[:, 0, :]
    )
    return zip_ap_2d(max, happy_paths, self_loops)

def forward_with_trace(model,batch_x,mask,dropout=0):
    transition_matrices = model.sopa.get_transition_matrices(batch_x,dropout)
    num_patterns = model.sopa.total_num_patterns
    n = batch_x.size()[2]
    batch_size=1 #Can only use 1 at a time
    doc_lens=torch.sum(mask,dim=1)
    
    scores = model.sopa.to_cuda(fixed_var(model.sopa.semiring.zero((batch_size, num_patterns))))
    
    eps_value = model.sopa.get_eps_value()
    eps_value=eps_value.cpu().detach()

    end_states = model.sopa.end_states
    
    def restart_padding(t):
        return [
            BackPointer(
                score=x,
                previous=None,
                transition=None,
                start_token_idx=t,
                end_token_idx=t
            )
            for x in model.sopa.semiring.one(num_patterns)
        ]
    def zero_padding(t):
        return [
            BackPointer(
                score=x,
                previous=None,
                transition=None,
                start_token_idx=t,
                end_token_idx=t
            )
            for x in model.sopa.semiring.zero(num_patterns)
        ]

    
    
    hiddens = model.sopa.to_cuda(Variable(model.sopa.semiring.zero(batch_size,
                                                        num_patterns,
                                                        model.sopa.max_pattern_length)))
    if model.sopa.deep_supervision:
        s_t = model.sopa.to_cuda(model.sopa.semiring.zero(n,num_patterns)).cpu()
        # convert s_t to back-pointers
        s_t = \
            [
                [
                    BackPointer(
                        score=pattern,
                        previous=None,
                        transition=None,
                        start_token_idx=0,
                        end_token_idx=0
                    )
                    for pattern in n
                ]
                for n in s_t
            ]
    else:
        scores = model.sopa.to_cuda(model.sopa.semiring.zero(batch_size,num_patterns)).cpu()
        # convert scores to back-pointers
        scores = \
                [
                    [
                        BackPointer(
                            score=scores_bp,
                            previous=None,
                            transition=None,
                            start_token_idx=0,
                            end_token_idx=0
                        )
                        for scores_bp in scores_patient
                    ]
                    for scores_patient in scores
                ]
        
    # set start state (0) to 1 for each pattern in each doc
    hiddens[:, :, 0] = model.sopa.to_cuda(model.sopa.semiring.one(batch_size, num_patterns))
    hiddens=hiddens.squeeze(0)
    hiddens=hiddens.cpu().detach()
    
    # convert to back-pointers
    hiddens = \
        [
            [
                BackPointer(
                    score=state_activation,
                    previous=None,
                    transition=None,
                    start_token_idx=0,
                    end_token_idx=0
                )
                for state_activation in pattern
            ]
            for pattern in hiddens
        ]
    # extract end-states
    end_states=model.sopa.end_states.squeeze(-1).cpu().detach()
    end_state_back_pointers = [
        bp[end_state]
        for bp, end_state in zip(hiddens, end_states)
    ]
    
    
    
    for token_idx, transition_matrix in enumerate(transition_matrices):
        transition_matrix=transition_matrix.squeeze(0)
        hiddens = transition_once_with_trace(model.sopa,
                                             token_idx,
                                             eps_value,
                                             hiddens,
                                             transition_matrix,
                                             restart_padding,
                                             zero_padding)
        
        # extract end-states and max with current bests
        end_state_back_pointers = [
            hidden_bps[end_state]
            for hidden_bps, end_state in zip(hiddens, end_states)
        ]    
        if model.sopa.deep_supervision:
            s_t[token_idx] = [
                                BackPointer(
                                    score=model.sopa.semiring.to_float(end_state_bp.score),
                                    previous=end_state_bp.previous,
                                    transition=end_state_bp.transition,
                                    start_token_idx=end_state_bp.start_token_idx,
                                    end_token_idx=end_state_bp.end_token_idx
                                )
                                for end_state_bp in end_state_back_pointers
                             ]  #Equation8b
        else:
            active_doc_idxs = torch.nonzero(torch.gt(doc_lens, token_idx)).squeeze().cpu()
            
            # extract end-states and max with current bests
            scores[active_doc_idxs]= [
                max(best_bp, hidden_bps[end_state])
                for best_bp, hidden_bps, end_state in zip(scores[active_doc_idxs], hiddens, end_states)
            ]
            
    if not model.sopa.deep_supervision:
        scores[active_doc_idxs]= [
                                    BackPointer(
                                                score=model.sopa.semiring.to_float(score_bp.score),
                                                previous=score_bp.previous,
                                                transition=score_bp.transition,
                                                start_token_idx=score_bp.start_token_idx,
                                                end_token_idx=score_bp.end_token_idx
                                    )
                                    for score_bp in scores[active_doc_idxs]
                                ]
    if not model.sopa.deep_supervision:
        #score dim is 1 x num_pattern
        return scores
    else:
        #S_t dim is seq_len x num_pattern
        return s_t
