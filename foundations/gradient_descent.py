class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        
        x_new = init

        itr = 0
        while itr < iterations:
            x_old = x_new

            x_new = x_old - learning_rate * 2 * x_old

            itr = itr + 1
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        pass

        return round(x_new,5)
