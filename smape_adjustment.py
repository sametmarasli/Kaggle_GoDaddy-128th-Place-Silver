import numpy as np


def smape(Y_predict, Y_test):
    result = np.linalg.norm(Y_predict - Y_test, axis = 1)
    result = np.abs(result)
    denom = np.linalg.norm(Y_predict, axis = 1)
    denom += np.linalg.norm(Y_test, axis = 1)
    result /= denom
    result *= 100 * 2
    result = np.mean(result)
    return result
epsilon = 1e-6
param_search = np.arange(10, 200, 20)

scores = []
for i in param_search:
    print(i)

    # definition of ztransformation.

    def ztransform1(Y, param=i):
        return 1 / (param + Y)

    # inverse transformation, Y = inverseZ(Z)

    def inverseZ1(Z, param=i):
        return -param + 1 / Z
    
    
    model = TransformedTargetRegressor(GradientBoostingRegressor(loss='squared_error', n_estimators=50,max_depth=10),func= ztransform1, inverse_func=inverseZ1)

    model.fit( train_X.loc['2022-05-01':'2022-09-01',['mdensity_lag1','mdensity_lag2','mdensity_lag3']], train_y.loc['2022-05-01':'2022-09-01',['target_0']]) 
        
    print(SMAPE_1(epsilon+model.predict(train_X.loc['2022-02-01':'2022-09-01',['mdensity_lag1','mdensity_lag2','mdensity_lag3']]),train_y.loc['2022-02-01':'2022-09-01',['target_0']].values))
    print(SMAPE_1(epsilon+model.predict(train_X.loc['2022-10-01',['mdensity_lag1','mdensity_lag2','mdensity_lag3']]),train_y.loc['2022-10-01',['target_0']].values))
    
# 160
# 1.3528178494715637
# 1.4203927419328115
# 190
# 1.3527584337361627
# 1.4161016190620148
