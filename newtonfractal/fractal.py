import os
from math import isnan, log
from glob import glob
from datetime import datetime
from time import perf_counter
from numpy import zeros, float32
from matplotlib.pyplot import show, figure, imshow, axis, savefig, close
from sympy import solve, solveset, diff, lambdify, simplify, sympify, re, im
from PIL.Image import open as PILopen

# THINGS I WANNA DO
# 01. Solve for z - f/diff(f) = 2x or 2yi (auto-bounds)         [Medium] [High] 1
# 02. Add symmetry detector (faster)                            [Hard]   [Med] -1
# 03. Add GPU support (faster)                                  [Hard]   [Med] -1
# 04. Make bloom/Red sin graphs/color by iterations (pretty)    [Hard]   [Med] -1
# 05. Improve time estimate (accuracy)                          [Easy]   [Low]  0
# 06. Difference between cycle and divergence                   [Medium] [Low] -1
# 07. Generalised form z - a * f/diff(f) + c (expand)           [Medium] [Low] -1
# 08. Create documentation
# 09. Average colors to stop too light or dark image
# 10. Examples
# 11. Unittest
# 12. __repr__
# 13. Image and timer update rather than new line
# 14. Add to Github
# 15. Custom color settings (see 04)

## Refactor/simple additions
# add bounds, smart default bounds?
# root limit for solveset
# flexible folders
# create and delete temp gif folder
# split things up further

class Fractal:
    def __init__(self, f="z**3-1", var="z"):
        # f(z)
        self.var = sympify(var)
        self.f = simplify(sympify(f))

        # f(z)/f'(z)
        self.f_over_f_dash = simplify(self.f/diff(self.f))

        # z - f(z)/f'(z)
        newton = simplify(self.var - self.f_over_f_dash)
        self.newton_lambd = lambdify(self.var, newton)

        # |z - f(z)/f'(z)| = |z| solutions
        self.mod_solution_0 = simplify(self.f_over_f_dash)
        self.mod_solution_z = simplify(2*self.var - self.f_over_f_dash)
        #self.mod_solution_x = simplify(2*x - self.f_over_f_dash)
        #self.mod_solution_y = simplify(2*y*1j - self.f_over_f_dash)

        # attributes
        self.TOLERANCE = 10**-6
        self.numerical_iterator = 20

        self.SAVE=False
        self.GIF=False
        self.CONTOUR=False
        self.RETAIN=False

        self.progress = None
        self.end_progress = None
        self.empty_value = None

        self.colors = [
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

    def root_solver(self, equation, bounds=[-10,10,-10,10],res=[100,100]): # add custom bounds, resolution
        roots = []

        roots += self.solve_with_sets(equation)
        
        if roots == []:
            roots += self.solve_with_algebra(equation)

        if roots == []:
            newton = self.var - equation/diff(equation)
            newton = lambdify(self.var,simplify(newton))

            step_x = (bounds[1]-bounds[0])/res[0]
            step_y = (bounds[3]-bounds[2])/res[1]
            for i in range(res[0] + 1):
                for j in range(res[1] + 1):
                    z = (i*step_x + bounds[0]) + (j*step_y + bounds[2])*1j
                    roots += self.solve_with_numerical(newton, z)
                percent_complete = 100*i/res[0]
                print(f"Finding roots numerically: {percent_complete}%")

        roots = list(set(roots))
        roots = [complex(root) for root in roots]

        return roots

    def solve_with_sets(self, equation, root_limit=10):
        roots = []
        counter = 0
        try:
            for root in solveset(equation):
                roots += [root]
                counter += 1
                if counter > root_limit:
                    break
            return roots
        except (NotImplementedError, TypeError):
            return []

    def solve_with_algebra(self, equation):
        try:
            roots = solve(equation)
            return roots
        except (NotImplementedError, TypeError):
            return []

    def solve_with_numerical(self, newton, z):
        initial_guess = z
        iterations = 0
        while iterations <= self.numerical_iterator:
            try:
                z = newton(z)
            except ZeroDivisionError:
                return []
            dif = z - initial_guess
            if abs(dif.real) < self.TOLERANCE and abs(dif.imag) < self.TOLERANCE:
                z_real = round(z.real,round(-log(self.TOLERANCE,10)))
                z_imag = round(z.imag,round(-log(self.TOLERANCE,10)))
                return [z_real + z_imag*1j]
            if isnan(z.real) or isnan(z.imag):
                pass
            elif abs(dif.real) > self.TOLERANCE**-1 or abs(dif.imag) > self.TOLERANCE**-1:
                return []
            iterations += 1
            initial_guess = z
        return []

    def create_colors(self, roots):
        while len(roots) > len(self.colors):
            more_colors = []
            for i in self.colors:
                templist = []
                for j in i:
                    templist.append(j/2)
                more_colors.append(templist)
            self.colors += more_colors

        colors = dict(zip(roots, self.colors))

        return colors

    def newton_method(self, z, roots, max_ite):
        iterations = 0
        while iterations <= max_ite:
            for root in roots:
                dif = z - root
                if abs(dif.real) < self.TOLERANCE and abs(dif.imag) < self.TOLERANCE:
                    return root, iterations
            try:
                z = self.newton_lambd(z)
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

    def create_images(self, cool_array, max_ite):
        figure()
        axis('off')
        imshow(cool_array)
        show()
        if self.SAVE is True and self.GIF is True:
            if self.RETAIN is False:
                file_format = ".png"
            else:
                file_format = ".png"
            savefig(r"C:/Users/camer/Documents/Fractal Images/GIF images/{0}".format(
                str(max_ite).zfill(3)) + file_format,
                dpi=1000,
                bbox_inches='tight')
            close()
            print("PNG for gif images saved")
        elif self.SAVE is True and self.GIF is False:
            savefig(r"C:/Users/camer/Documents/Fractal Images/Fractal Pretty {0}".format(
                str(datetime.now()).replace(':',' ')) + ".png",
                dpi=max(cool_array.shape)/3,
                bbox_inches='tight')
            close()
            print("PNG image(s) saved")
        else:
            print("Image not saved.")

    def make_gif(self, frame_folder,file_format=".png"):
        frames = [PILopen(image) for image in glob(f"{frame_folder}/*{file_format}")]
        frame_one = frames[0]
        frame_one.save(
            "C:/Users/camer/Documents/Fractal Images/GIFs/fractal gif {0}.gif".format(
                str(datetime.now()).replace(':',' ')),
            format="GIF", append_images=frames,
                   save_all=True, duration=100, loop=0)

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

        roots_f = self.root_solver(self.f)

        roots_mod = []
        roots_mod += self.root_solver(self.mod_solution_0)
        roots_mod += self.root_solver(self.mod_solution_z)

        if roots_f+roots_mod == []:
            print("No roots found. Try increasing numerical_iterator")
            return None

        biggest_root = max([abs(root) for root in roots_f+roots_mod])
        self.empty_value = complex(1 + biggest_root)

        colors = self.create_colors(roots_f)

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

        for max_ite in range(ITE_BEGIN, end_point + 1):
            for i in range(RES):
                time_begin = perf_counter()
                for j in range(RES):
                    if (cool_array[i][j] == zeros(3)).all() or self.CONTOUR:
                        z = min_x + j*step_x + (min_y + i*step_y)*1j
                        root, ites = self.newton_method(z, roots_f, max_ite)
                        if self.CONTOUR:
                            contour_bands = max_ite
                        if root != self.empty_value:
                            color_multiplier = 1 - (1+ites)/(1+contour_bands)
                            cool_array[i][j] = [rgb*color_multiplier for rgb in colors[root]]
                        for root in roots_mod:
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
                      f": {int(100*progress_percent)}%")
                
                #check modulo maths
                if 1000*progress_percent % int(10000/RES) < 10*RES/self.end_progress and RES <= 10000:
                    if self.SAVE == True:
                        temp_SAVE = True
                        self.SAVE = False
                    else:
                        temp_SAVE = False
                    self.create_images(cool_array, max_ite)

            if temp_SAVE == True:
                temp_SAVE = False
                self.SAVE = True
            self.create_images(cool_array,max_ite)

        if maximal_ites[-1] == end_point:
            print(f"Maximum iterations required for converged points was {maximal_ites[-2]}")
        else:
            print(f"Maximum iterations required for converged points was {maximal_ites[-1]}")
        toc = perf_counter()
        print(toc-tic)
        
        return cool_array
