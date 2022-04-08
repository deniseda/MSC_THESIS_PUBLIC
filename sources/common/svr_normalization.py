from sklearn.metrics import r2_score



# models --> svr models previously saved
# list_metabolites --> QCs or samples peaks
# injection_order --> injection order of samples or QCs. 
# Depends on what you want predict 

def prediction(models, list_metabolites, injection_order):
    meta_normalized = []
    for i in range(len(list_metabolites)):
        meta_normalized.append(models[i].predict(injection_order.reshape(-1,1)))
    return meta_normalized


# Y'
def compute_ratio(original, predicted):
    ratio = []
    for i in range(0,len(original)):
        metab = []
        for j in range(0, len(original[i])):
            metab.append(original[i][j]/ predicted[i][j])
        ratio.append(metab)
    return ratio



def ratio_to_intensity(ratio, rawdata):
    intensity = []
    for r in range(0, len(ratio)):
        intensity.append(ratio[r] * rawdata[r])
    return intensity



def score_R2(realvalue, predictvalue):
    score_kernel = []
    for i in range(0, len(realvalue)):
        score = []
        score.append(r2_score(realvalue[i], predictvalue[i]))
        for s in score:
            score_kernel.append(s)
    return score_kernel