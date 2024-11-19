"""Temp module docstring"""

from math import isnan, log
from glob import glob
from datetime import datetime
from time import perf_counter
from numpy import zeros, float32
from matplotlib.pyplot import pause, subplots, show, figure, imshow, axis, savefig, close
from sympy import solve, solveset, diff, lambdify, simplify, EmptySet
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, \
    implicit_multiplication_application
from PIL.Image import open as PILopen

##THINGS I WANNA DO
# Solve for z - f/diff(f) = 2x or 2yi (auto-bounds)      [Medium] [High] 1
# Convert operations to arrays                           [Medium] [High] 1
# Improve time estimate (accuracy)                       [Easy]   [Low]  0
# Examples                                               [Easy]   [Low]  0
# Unittest                                               [Easy]   [Low]  0
# __repr__                                               [Easy]   [Low]  0
# Custom color settings                                  [Hard]   [High] 0
# Add symmetry detector (faster)                         [Hard]   [Med] -1
# Add GPU support (faster)                               [Hard]   [Med] -1
# Difference between cycle and divergence                [Medium] [Low] -1
# Generalised form z - a * f/diff(f) + c (expand)        [Medium] [Low] -1
# Create documentation                                   [Medium] [Low] -1
# Average colors to stop too light or dark image         [Medium] [Low] -1
# Zoomable with recalculation                            [Hard]   [Low] -2

# git add .
# git commit -m "message"
# git push origin main

class Fractal:
    
    
    def __init__(self, f="z**3-1"):
        self.numerical_iterator = 20
        self.tol = 10**-6
        
        self.f = f

        self.CONTOUR=False

        self.progress = None
        self.end_progress = None
        self.empty_value = None
    
    @property
    def f(self):
        return self._f
    
    @f.setter 
    def f(self, f):
        """
        This turns the class input from a string to a 
        symbolic expression.
        Then, it calls root_solver to calculate the roots of
        the expression and its auto-bounds function.
        
        """
        transformations = (
            standard_transformations
            + (implicit_multiplication_application,)
            )
        f = parse_expr(f, transformations=transformations)
        self._f = simplify(f)
        self._var = list(f.free_symbols)[0]

        f_over_f_dash = self.f / diff(self.f)
        f_over_f_dash = simplify(f_over_f_dash)
        newton = self._var - f_over_f_dash
        newton = simplify(newton)
        self.newton_lambda = lambdify(self._var, newton)

        # |z - f(z)/f'(z)| = |z| solutions
        # Used to calculate automatic image boundaries
        mod_solution_0 = simplify(f_over_f_dash)
        mod_solution_z = simplify(2*self._var - f_over_f_dash)
        # mod_solution_x = simplify(2*x - self._f_over_f_dash)
        # mod_solution_y = simplify(2*y*1j - self._f_over_f_dash)
        
        self._roots_f = self._root_solver(self.f)

        self._roots_mod = []
        self._roots_mod += self._root_solver(mod_solution_0)
        self._roots_mod += self._root_solver(mod_solution_z)

        if self._roots_f == []:
            print("No roots found. Try increasing numerical_iterator")

    def _root_solver(self, equation, bounds=[-10,10,-10,10],res=[100,100]): # add custom bounds, resolution
        
        roots = self._solve_with_sets(equation)

        if roots == 'No roots found':
            roots = self._solve_with_algebra(equation)

        if roots == 'No roots found':
            roots = []
            newton = self._var - equation/diff(equation)
            newton = lambdify(self._var,simplify(newton))

            step_x = (bounds[1]-bounds[0])/res[0]
            step_y = (bounds[3]-bounds[2])/res[1]
            for i in range(res[0] + 1):
                for j in range(res[1] + 1):
                    z = (i*step_x + bounds[0]) + (j*step_y + bounds[2])*1j
                    roots += self.solve_with_numerical(newton, z)
                percent_complete = 100*i/res[0]
                print(f"Finding roots numerically: {percent_complete}%", 
                      end="\r")

        roots = list(set(roots))
        roots = [complex(root) for root in roots]

        return roots

    def _solve_with_sets(self, equation, root_limit=10):
        """
        This tries to solve equation=0 using sympy.solveset.
        root_solver first call this to solve the equation since solveset
        can return all roots, including an infinite number of roots, as
        a set. For example, this can solve 'sin(z)=0'.
        If solveset fails (NotImplementedError) or it returns a condition 
        instead of a set of numbers (TypeError), this returns an empty
        list.
        
        """
        try:
            roots_set = solveset(equation)
            for _ in roots_set:
                pass
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
        This tries to solve equation=0 using sympy.solve.
        root_solver calls this after solve_with_sets as solveset is not
        always a better version of solve. solve is older and hence more
        robust so it can succeed in certain cases where solveset fails.
        For example, z**z=0 has no solutions. solveset returns a 
        ConditionSet that just restates the equation. solve returns an
        empty list which is what we want.

        """
        try:
            roots = solve(equation)
        except (NotImplementedError):
            roots = 'No roots found'
        return roots

    def solve_with_numerical(self, newton, z):
        initial_guess = z
        iterations = 0
        while iterations <= self.numerical_iterator:
            try:
                z = newton(z)
            except ZeroDivisionError:
                return []
            
            dif = z - initial_guess
            if abs(dif.real) < self.tol and abs(dif.imag) < self.tol:
                z_real = round(z.real,round(-log(self.tol,10)))
                z_imag = round(z.imag,round(-log(self.tol,10)))
                return [z_real + z_imag*1j]
            if isnan(z.real) or isnan(z.imag):
                pass
            elif abs(dif.real) > self.tol**-1 or abs(dif.imag) > self.tol**-1:
                return []
            iterations += 1
            initial_guess = z
        return []

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

    def newton_method(self, z, roots, max_ite):
        iterations = 0
        while iterations <= max_ite:
            for root in roots:
                dif = z - root
                if abs(dif.real) < self.tol and abs(dif.imag) < self.tol:
                    return root, iterations
            try:
                z = self.newton_lambda(z)
            except (ZeroDivisionError, OverflowError):
                return self.empty_value, iterations
            iterations += 1

        return self.empty_value, iterations

    def time_estimator(self, previous_time_estimate, time_begin, time_end):
        progress_percent = (1+self.progress)/self.end_progress

        time_estimate = self.end_progress - self.progress
        time_estimate *= time_end - time_begin

        if previous_time_estimate < time_estimate < 1.5*previous_time_estimate:
            return previous_time_estimate, progress_percent

        return time_estimate, progress_percent

    def _create_images(self, fig, im, cool_array, max_ite):
        #use %matplotlib to get interactive plot
        im.set_data(cool_array)
        fig.canvas.draw_idle()
        pause(0.001)

    def save(self, output_folder, cool_array, colors='default'):
        figure()
        axis('off')
        imshow(cool_array)
        show()
        savefig(f"{output_folder} {0}".format(
            str(datetime.now()).replace(':',' ')) + ".png",
            dpi=max(cool_array.shape)/3,
            bbox_inches='tight')
        close()
        print("PNG image(s) saved")
        
# =============================================================================
#         savefig(r"C:/Users/camer/Documents/Fractal Images/GIF images/{0}".format(
#             str(max_ite).zfill(3)) + file_format,
#             dpi=1000,
#             bbox_inches='tight')
# =============================================================================

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

    def art(self, RES=200, ITE_BEGIN=20, ITE_END=1, bounds=[None,None,None,None]):
        tic = perf_counter()

        self.progress = 0
        time_estimate = 0
        maximal_ites = [0]

        if ITE_END < ITE_BEGIN:
            self.end_progress = RES**2
            end_point = ITE_BEGIN
        else:
            self.end_progress = (ITE_END+1 - ITE_BEGIN)*RES**2
            end_point = ITE_END

        contour_bands = end_point

        cool_array = zeros([RES,RES,3], dtype=float32)

        e = 0*4/RES # error around pink circles

        biggest_root = max(abs(root) for root in self._roots_f+self._roots_mod)
        self.empty_value = complex(1 + biggest_root)

        colors = self._create_colors(self._roots_f)

        if [i==None for i in bounds] == 4*[False]:
            min_x = bounds[0]
            max_x = bounds[1]
            min_y = bounds[2]
            max_y = bounds[3]
        else:
            a = self.empty_value - 1
            min_x = -a
            max_x =  a
            min_y = -a
            max_y =  a

        step_x = (max_x-min_x)/RES
        step_y = (max_y-min_y)/RES
        
        fig, ax = subplots(1,1)
        im = ax.imshow(cool_array)
        axis('off')

        for max_ite in range(ITE_BEGIN, end_point + 1):
            for i in range(RES):
                time_begin = perf_counter()
                for j in range(RES):
                    if (cool_array[i][j] == zeros(3)).all() or self.CONTOUR:
                        z = min_x + j*step_x + (min_y + i*step_y)*1j
                        root, ites = self.newton_method(z, self._roots_f, max_ite)
                        if self.CONTOUR:
                            contour_bands = max_ite
                        if root != self.empty_value:
                            color_multiplier = 1 - (1+ites)/(1+contour_bands)
                            cool_array[i][j] = [rgb*color_multiplier for rgb in colors[root]]
                        for root in self._roots_mod:
                            if e != 0 and - e < abs(z)-abs(root) < e:
                                cool_array[i][j] = [249/256, 158/256, 220/256]
                        if  maximal_ites[-1] < ites <= end_point:
                            maximal_ites.append(ites)

                time_end = perf_counter()
                self.progress += RES
                time_estimate, progress_percent = self.time_estimator(time_estimate*RES, time_begin, time_end)
                time_estimate = time_estimate/RES
                
                print(f"{str(int(time_estimate)).zfill(4)}s :" + \
                      (1 + int(10*progress_percent)) * "=" + \
                      (10 - int(10*progress_percent)) * " " +  \
                      f": {int(100*progress_percent)}%", end="\r")
                
                #check modulo maths
                if 1000*progress_percent % int(10000/RES) < 10*RES/self.end_progress and RES <= 10000:
                    self._create_images(fig, im, cool_array, max_ite)

        if maximal_ites[-1] == end_point:
            print(f"Maximum iterations required for converged points was {maximal_ites[-2]}")
        else:
            print(f"Maximum iterations required for converged points was {maximal_ites[-1]}")
        toc = perf_counter()
        print(f"Time taken in seconds: {toc-tic}")
        
        return cool_array
