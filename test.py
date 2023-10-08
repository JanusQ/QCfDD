from scipy.optimize import minimize

res=minimize(lambda x: x, x0=[1], bounds=[(0, 2)], options={"maxiter": 2})
print(res.x)
print(res.fun)
