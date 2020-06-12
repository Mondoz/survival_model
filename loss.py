""" losses for training neural networks """
from mxnet import ndarray
from mxnet.base import numeric_types
from mxnet.gluon.block import HybridBlock
import numpy as np

class Survial_loss(HybridBlock):
    def __init__(self, **kwargs):
        super(Survial_loss, self).__init__(**kwargs)
    
    #def hybrid_forward(self, F, y_true, y_pred, train_data):
    def hybrid_forward(self, F, y_true, y_pred):
        y_pred = F.L2Normalization(y_pred, mode='instance')

        hazard_ratio = F.exp(y_pred)
        cumsum_hazard_ratio  = ndarray.array(np.cumsum(hazard_ratio.asnumpy())).as_in_context(y_pred.context)
        log_risk = F.log(cumsum_hazard_ratio)
        uncensored_likelihood = y_pred.T - log_risk   
        censored_likelihood =  uncensored_likelihood * y_true
        logL = - F.sum(censored_likelihood)
        observations = F.sum(y_true, axis = 0)
        return logL / (observations+1e-10) 
        #return censored_likelihood
        '''
        
        logL = 0
        # pre-calculate cumsum
        y_pred = F.L2Normalization(y_pred, mode='instance')

        cumsum_y_pred = ndarray.array(np.cumsum(y_pred.asnumpy())).as_in_context(y_pred.context)
        hazard_ratio = F.exp(y_pred)
        cumsum_hazard_ratio = ndarray.array(np.cumsum(hazard_ratio.asnumpy())).as_in_context(y_pred.context)
        
        if train_data['ties'] == 'noties':
            log_risk = F.log(cumsum_hazard_ratio)
            likelihood = y_pred.T - log_risk
            # dimension for E: np.array -> [None, 1]
            uncensored_likelihood = likelihood * y_true
            logL = -F.sum(uncensored_likelihood)
        else:
            # Loop for death times
            for t in train_data['failures']:
                                                                       
                tfail = train_data['failures'][t]
                trisk = train_data['atrisk'][t]
                d = len(tfail)
                dr = len(trisk)

                logL = logL - cumsum_y_pred[tfail[-1]] + (0 if tfail[0] == 0 else cumsum_y_pred[tfail[0]-1])

                if train_data['ties'] == 'breslow':
                    s = cumsum_hazard_ratio[trisk[-1]]
                    logL = logL + F.log(s) * d
                elif train_data['ties'] == 'efron':
                    s = cumsum_hazard_ratio[trisk[-1]]
                    r = cumsum_hazard_ratio[tfail[-1]] - (0 if tfail[0] == 0 else cumsum_hazard_ratio[tfail[0]-1])
                    for j in range(d):
                        logL = logL + F.log(s - j * r / d)
                else:
                    raise NotImplementedError('tie breaking method not recognized')
        # negative average log-likelihood
        
        observations = F.sum(y_true, axis = 0)
        return logL / (observations+1) 
        '''