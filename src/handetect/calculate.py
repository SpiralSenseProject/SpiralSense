from scipy.optimize import linprog

# Coefficients for the objective function (negative because linprog does minimization)
c = [-0.88, -0.88, -0.85]

# Coefficients for the inequality constraint (sum of weights = 1)
A = [[1, 1, 1]]
b = [1]

# Bounds for each weight (between 0 and 1)
bounds = [(0, 1), (0, 1), (0, 1)]

# Solve the linear programming problem
result = linprog(c, A_eq=A, b_eq=b, bounds=bounds)

# The optimal weights
optimal_weights = result.x
print("Optimal weights:", optimal_weights)