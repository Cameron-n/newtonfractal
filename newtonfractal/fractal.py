"""Temp module docstring"""

from math import log
from glob import glob
from datetime import datetime
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import get_backend
from PIL.Image import open as PILopen

from sympy import solve, solveset, diff, lambdify, simplify, EmptySet
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, \
    implicit_multiplication_application

##THINGS I WANNA DO
# Solve for z - f/diff(f) = 2x or 2yi (auto-bounds)      [Medium] [High] M
# NumPy support that actually works...                   [Medium] [High] 1
# Custom color settings                                  [Hard]   [High] 0
# Improve time estimate (accuracy)                       [Easy]   [Low]  0
# Examples                                               [Easy]   [Low]  0
# Unittest                                               [Easy]   [Low]  0
# __repr__                                               [Easy]   [Low]  0
# Add symmetry detector (faster)                         [Hard]   [Med] -1
# Add GPU support (faster)                               [Hard]   [Med] -1
# Difference between cycle and divergence                [Medium] [Low]  M
# Generalised form z - a * f/diff(f) + c (expand)        [Medium] [Low] -1
# Create documentation                                   [Medium] [Low] -1
# Zoomable with recalculation                            [Hard]   [Low] -2

# git add .
# git commit -m "message"
# git push origin main

class Fractal:
    """
    The documentation will refer to the input function as f(z).
    
    """
    
    def __init__(
            self, 
            f="z**3-1", 
            max_iter=20, 
            tol=10**-6, 
            bounds=[-10, 10, -10, 10], 
            res=[100, 100],
        ):
        
        if get_backend() == "inline":
            print("""
                  Please use the '%matplotlib' magic command. The inline 
                  backend will cause the images to display improperly.
                  """)
        
        self.max_iter = max_iter
        self.tol = tol
        self.bounds = bounds
        self.res = res
        self.decimals = round(-log(tol,10))
        # f is defined here as its setter function
        # relys on the above variables.
        self.f = f

    @property
    def roots(self):
        return self._roots_f, self._roots_mod
    
    @property
    def f(self):
        return self._f
    
    @f.setter 
    def f(self, f):
        """
        Turns the class input from a string to a 
        symbolic expression.
        Then, it calls root_solver to calculate the roots of
        the expression and its auto-bounds functions.
        
        """
        
        # Parses input string.
        transformations = (
            standard_transformations
            + (implicit_multiplication_application,)
            )
        f = parse_expr(f, transformations=transformations)
        
        self._f = simplify(f)
        self._z = list(f.free_symbols)[0]
        
        # Finds roots of f(z).
        self._roots_f = self._root_solver(self.f)
        if self._roots_f == []:
            raise Exception("No roots found.")

        # |z - f(z)/f'(z)| = |z| solutions. 
        # ----------------------------------------------------
        # This is used to calculate automatic image boundaries. 
        # See the docs for a mathematical explanation.
        
        self._roots_mod = []
        
        mod_solution_0 = simplify(self.f / diff(self.f))
        self._roots_mod += self._root_solver(mod_solution_0)
        
        mod_solution_z = simplify(2*self._z - simplify(self.f / diff(self.f)))
        self._roots_mod += self._root_solver(mod_solution_z)

    def _root_solver(self, equation):
        """
        Calculates the values of z for equation(z) = 0, i.e. 
        the roots. It attempts to solve this with three methods:
            1. using sympy.solvesets
            2. using sympy.solve
            3. numerically using newtons method
        Note that we use newtons method as a numerical solver as
        if it can't find the roots then it won't be able to find
        the roots later on for drawing the fractal.
        Also note that 'No roots found' is distinct from an empty
        list. The former simply means no roots have yet been found,
        the later means there are no roots.
        """

        roots = self._solve_with_numerical(equation)

        roots = list(set(roots))
        roots = [complex(root) for root in roots]
        roots = [
            round(z.real,round(-log(self.tol,10))) 
            + round(z.imag,round(-log(self.tol,10))) *1j
            for z in roots]

        return roots

    def _solve_with_sets(self, equation, root_limit):
        """
        [OUTDATED, MAY BE REMOVED]
        Tries to solve equation=0 using sympy.solveset.
        root_solver first call this to solve the equation since solveset
        can return all roots, including an infinite number of roots, as
        a set. For example, this can solve 'sin(z)=0'.
        If solveset fails (NotImplementedError) or it returns a condition 
        instead of a set of numbers (TypeError), this returns an empty
        list.
        
        """
        try:
            roots_set = solveset(equation)
            counter = 0
            for _ in roots_set:
                counter += 1
                if counter > 1:
                    break
        except (NotImplementedError, TypeError):
            roots_set = set()
        
        roots = []
        counter = 0
        for root in roots_set:
            roots += [root]
            counter += 1
            if counter > root_limit:
                break
            
        if roots_set == EmptySet:
            roots = []
        elif roots == []:
            roots = 'No roots found'
        return roots

    def _solve_with_algebra(self, equation):
        """
        [OUTDATED, MAY BE REMOVED]
        Tries to solve equation=0 using sympy.solve.
        root_solver calls this after solve_with_sets as solveset is not
        always a better version of solve. solve is older and hence more
        robust so it can succeed in certain cases where solveset fails.
        For example, z**z=0 has no solutions. solveset returns a 
        ConditionSet that just restates the equation. solve returns an
        empty list which is what we want.

        """
        try:
            roots = solve(equation)
        except NotImplementedError:
            roots = 'No roots found'
        return roots
    
    def _solve_with_numerical(self, equation):
        """
        Tries to solve equation=0 using newtons method.
        This is a fallback method that should work for most sane
        functions. It effectively runs the same calculations used to 
        create a newton fractal, just fewer of them. The defaults
        go over a grid that's only 100x100.
        Note: this option may be less efficient than finding the 
        roots and adding extra colors dynamically. However, this may 
        still be useful for root finding before a full newtons method 
        is employed.
        
        """
        # z - f(z)/f'(z)
        newton = self._z - simplify(equation/diff(equation))
        newton = lambdify(self._z, simplify(newton))
        
        min_x, max_x, min_y, max_y = self.bounds
        step_x, step_y = (max_x - min_x)/self.res[0], (max_y - min_y)/self.res[1]
        
        self.empty_value = np.nan
        
        roots = []
        for y in range(self.res[0]):
            for x in range(self.res[1]):
                z = min_x + x*step_x + (min_y + y*step_y)*1j
                root, _ = self.newton_method(newton, z, self.tol, self.max_iter)
                if np.isnan(root) == False:
                    roots.append(root)
        
        return roots
    
    def newton_method(self, newton, z, tol, max_ite):
        """
        Newtons method itself. Calculates:
            z_n+1 = z_n - f(z)/f'(z)
        If the value calculated is within a tolerance, we return that
        as a root. Otherwise, we return the empty_value, i.e. the value
        that means we found no root and so no color needs to be drawn.
        We also track the number of iterations to estimate the time left
        until completion.
        
        """
        iterations = 0
        while iterations <= max_ite:
            z_previous = z
            try:
                z = newton(z)
            except (ZeroDivisionError, OverflowError):
                break
            dif = z - z_previous
            if abs(dif) < tol:
                root = round(z.real,self.decimals) + round(z.imag,self.decimals)*1j
                return root, iterations
            elif abs(dif) > tol**-1:
                break
            iterations += 1
            
        return self.empty_value, iterations

    def time_estimator(self, res, previous_time_estimate, time_begin, time_end):
        self.progress += res[0]
        progress_percent = (1 + self.progress)/self.end_progress

        time_estimate = self.end_progress - self.progress
        time_estimate *= time_end - time_begin

        if previous_time_estimate < time_estimate < 1.5*previous_time_estimate:
            time_estimate = previous_time_estimate

        time_estimate = time_estimate/res[0]
        
        print(f"{str(int(time_estimate)).zfill(4)}s :" + \
              (1 + int(10*progress_percent)) * "=" + \
              (10 - int(10*progress_percent)) * " " +  \
              f": {int(100*progress_percent)}%", end="\r")
            
        return time_estimate, progress_percent

    def timer(func):
        """
        Simple decorator to time the execution of a function.
        
        """
        def inner(*arg, **kwarg):
            tic = perf_counter()
            output = func(*arg, **kwarg)
            toc = perf_counter()
            print(f"Time taken in seconds: {toc-tic}")
            return output
        return inner
    
    @timer
    def art(
            self, 
            max_iter=20,
            tol=10**-6, 
            bounds=[None, None, None, None],
            res=[200,200],
        ):
        
        # image setup
        cool_array = np.zeros([res[0],res[1],3], dtype=np.float32)
        fig, ax = plt.subplots(1,1)
        im = ax.imshow(cool_array)
        plt.axis('off')
        
        # time estimate setup
        self.progress = 0
        time_estimate = 0
        self.end_progress = res[0]*res[1]
        
        # colors creation
        colors = self._create_colors(self._roots_f)

        # bounds, move to root solver
        if [i is None for i in bounds] == 4*[False]:
            min_x, max_x, min_y, max_y = bounds
            self.empty_value = min_x - 1
        else:
            biggest_root = max(abs(root) for root in self._roots_f+self._roots_mod)
            self.empty_value = complex(1 + biggest_root)
            a = self.empty_value - 1
            min_x, max_x, min_y, max_y = [-a, a, -a, a]
        
        # newton method setup
        maximal_ites = [0]
        self.decimals = round(-log(tol,10))
        newton = self._z - simplify(self.f/diff(self.f))
        newton = lambdify(self._z, simplify(newton))
        
        step_x, step_y = (max_x - min_x)/res[0], (max_y - min_y)/res[1]
        
        BLACK = np.zeros(3)
        for y in range(res[0]):
            time_begin = perf_counter() # start counter
            for x in range(res[1]):
                # put into seperate function
                if (cool_array[y][x] == BLACK).all():
                    z = min_x + x*step_x + (min_y + y*step_y)*1j
                    root, ites = self.newton_method(newton, z, tol, max_iter)
                    if root != self.empty_value:
                        color_multiplier = 1 - (1 + ites)/(1 + max_iter)
                        try:
                            cool_array[y][x] = [rgb*color_multiplier for rgb in colors[root]]
                        except KeyError:
                            # Will add dynamic colors here
                            pass
                        if maximal_ites[-1] < ites:
                            maximal_ites.append(ites)
                # function putting ends
            time_end = perf_counter()   # stop counter
            time_estimate, progress_percent = self.time_estimator(
                res, time_estimate*res[0], time_begin, time_end
                )
            
            #check modulo maths
            if 1000*progress_percent % int(10000/res[0]) < 10*res[0]/self.end_progress and res[0] <= 10000:
                self._create_images(fig, im, cool_array)   
        
        print("\n" + f"Maximum iterations required for converged points was {maximal_ites[-1]}")
    
        return cool_array
    
    def colors(self):
        DEFAULT_COLORS = [
            [1.0, 0.0, 0.0], #Red
            [0.0, 1.0, 0.0], #Green
            [0.0, 0.0, 1.0], #Blue
            [1.0, 1.0, 0.0], #Yellow
            [1.0, 0.0, 1.0], #Magenta
            [0.0, 1.0, 1.0], #Cyan
            [1.0, 0.5, 0.5],
            [0.5, 1.0, 0.5],
            [0.5, 0.5, 1.0],
            [1.0, 1.0, 0.5],
            [1.0, 0.5, 1.0],
            [0.5, 1.0, 1.0],
            ]
        return DEFAULT_COLORS
    
    def _create_colors(self, roots):
        # Creates more colors if there are more roots than colors
        colors = self.colors()
        
        while len(roots) > len(colors):
            more_colors = []
            for i in colors:
                templist = []
                for j in i:
                    templist.append(j/2)
                more_colors.append(templist)
            colors += more_colors

        colors = dict(zip(roots, colors))

        return colors
    
    def _create_images(self, fig, im, cool_array):
        """
        Updates a given figure with a given array. Used to dynamically
        update the plot when creating the newton fractal. Can also be
        called independantly if an array is created elsewhere.
        Does not function correctly with the inline backend.
        
        """
        #use %matplotlib to get plot that updates
        im.set_data(cool_array)
        fig.canvas.draw_idle()
        plt.pause(0.001)
        
    def save(self, output_folder, cool_array, colors='default'):
        fig, ax = plt.subplots(1,1)
        plt.axis('off')
        plt.imshow(cool_array)
        plt.show()
        name = str(datetime.now()).replace(':',' ')
        plt.savefig(f"{output_folder} {name}" + ".png",
            dpi=max(cool_array.shape)/3,
            bbox_inches='tight')
        plt.close()
        print("PNG image(s) saved")
        
    def gif(self, input_images, output_images=None, 
            duration=100, loop=0, file_format=".png"):
        
        if output_images is None:
            output_images = input_images
        frames = [PILopen(image) for image in glob(f"{input_images}/*{file_format}")]
        frame_one = frames[0]
        frame_one.save(
            output_images.format(
                str(datetime.now()).replace(':',' ')),
            format="GIF", append_images=frames,
            save_all=True, duration=duration, loop=loop)
