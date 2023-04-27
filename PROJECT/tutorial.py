from sympy import Symbol, Matrix, lambdify,symbols
import numpy as np
from matplotlib import pyplot as plt


def calc_lagrange_polynomial(t, x):

    z = symbols('t')
    lag_poly = []

    for i in range(len(x)):
        temp_poly = 1
        for k in range(0, len(x), 1):
            if k != i:
                temp_poly = temp_poly * ((z - t[k])/(t[i] - t[k]))
        lag_poly.append(temp_poly)

    total_poly = 0
    i = 0
    for polynom in lag_poly:
        f_part = polynom * x[i]
        i += 1
        total_poly = total_poly + f_part

    return total_poly


def calc_lagrange_value(lagrange_eqn, t_list):
    lag_val_list1 = []
    lag_val_list2 = []
    for time_list in t_list:
        temp_val_list1 = []
        temp_val_list2 = []
        for value in time_list:
            temp_val_list1.append(lagrange_eqn.subs({'t': value}))
            temp_val_list2.append((lagrange_eqn.diff('t')).subs({'t': value}) * 0.6818)
        lag_val_list1.append(temp_val_list1)
        lag_val_list2.append(temp_val_list2)
    return [lag_val_list1, lag_val_list2]


def calc_lagrange_val_at_time(lagrange_eqn, at_time):
    lag_pos = lagrange_eqn.subs({'t': at_time})
    lag_speed = (lagrange_eqn.diff('t')).subs({'t': at_time}) * 0.6818
    print(f"As per lagrange interpolation, position at {at_time} s is  {lag_pos} and speed is {lag_speed}")


def calc_cs_coefficients(t, x, x_dot, spline_type):

    h = []
    a = []
    b1 = []
    b = []
    d = []
    z = 0
    while z < len(x) - 1:
        h.append(t[z+1] - t[z])
        a.append(x[z])
        z += 1
        if z == len(x) - 1:
            a.append(x[z])

    z = 0

    while z < len(x):
        if z == 0:
            if spline_type == 't':
                b_0 = (3/h[z]) * (a[z+1] - a[z]) - (3 * x_dot[z])
                b1.append(b_0)
            else:
                b1.append(0)
        elif 0 < z < len(x) - 1:
            b_int = (3 / h[z]) * (a[z + 1] - a[z]) - (3 / h[z - 1]) * (a[z] - a[z - 1])
            b1.append(b_int)

        z += 1
        if z == len(x) - 1:
            if spline_type == 't':
                b_n = (3 * x_dot[z]) - (3 / h[z - 1]) * (a[z] - a[z - 1])
                b1.append(b_n)
            else:
                b1.append(0)

    b_vector = np.array(b1)

    z = 0

    square_a = np.zeros([len(x), len(x)])
    square_a_matrix = np.array(square_a)

    while z < len(x):
        i = z + 1

        if z == 0:
            if spline_type == 't':
                square_a_matrix[z][i] = h[z]
                square_a_matrix[z][i-1] = 2 * h[z]
            else:
                square_a_matrix[z][i] = 0
                square_a_matrix[z][i - 1] = 1

        elif 0 < z < len(x) - 1:
            while i >= z - 1:
                if i == z + 1:
                    square_a_matrix[z][i] = h[z]
                elif i == z - 1:
                    if i >= 0:
                        square_a_matrix[z][i] = h[z-1]
                else:
                    square_a_matrix[z][i] = 2 * (h[z] + h[z-1])
                i -= 1

        z += 1

        if z == len(x) - 1:
            i = z
            if spline_type == 't':
                square_a_matrix[z][i] = 2 * h[z-1]
                square_a_matrix[z][i - 1] = h[z-1]
            else:
                square_a_matrix[z][i] = 1
                square_a_matrix[z][i - 1] = 0

    x_vector = np.linalg.solve(square_a_matrix, b_vector)
    c = np.reshape(x_vector, -1)

    z = 0

    while z < len(x) - 1:
        if z < len(x) - 1:
            b_z = ((a[z+1] - a[z])/h[z]) - ((h[z]/3) * (2*x_vector[z] + x_vector[z+1]))
            b.append(b_z)
            d_z = (x_vector[z+1] - x_vector[z])/(3 * h[z])
            d.append(d_z)
        z += 1

    return [a, b, c, d]


def calc_pos_speed(t_list, cs_eq_list, dcs_eq_list):

    cs_val_list = []
    dcs_val_list = []

    for k in range(len(t_list)):
        temp_cs_list = []
        temp_dcs_list = []
        for num in t_list[k]:
            temp_cs_list.append(float(cs_eq_list[k].subs({y: num})))
            temp_dcs_list.append(float(dcs_eq_list[k].subs({y: num})) * 0.6818)
        cs_val_list.append(temp_cs_list)
        dcs_val_list.append(temp_dcs_list)

    min_speed = float(100)
    max_speed = float(-100)
    min_speed_time = 0
    max_speed_time = 0
    length = len(t_list[0])
    for i in range(len(t_list)):
        for k in range(length):
            if dcs_val_list[i][k] < min_speed:
                min_speed = dcs_val_list[i][k]
                min_speed_time = t_list[i][k]
            if dcs_val_list[i][k] > max_speed:
                max_speed = dcs_val_list[i][k]
                max_speed_time = t_list[i][k]

    return [cs_val_list, dcs_val_list, min_speed, min_speed_time, max_speed, max_speed_time]


def calc_pos_speed_at_time(t_list, cs_eq_list, dcs_eq_list, at_time):

    if at_time < t_list[0][0] or at_time > t_list[len(t_list)-1][len(t_list[0]) - 1]:
        print("Input time out of bounds!")
    else:
        for i in range(len(t_list)):
            if t_list[i][0] < at_time < t_list[i][len(t_list[0])-1]:
                position = float(cs_eq_list[i].subs({y: at_time}))
                speed = float(dcs_eq_list[i].subs({y: at_time})) * 0.6818
                print(f"As per cubic spline interpolation,the position and speed of the vehicle at {at_time} s is "
                      f"{position} feet and {speed} MPH.")
                break


def plot_graph(t_list, cs_val_list, dcs_val_list, lag_pos_val, lag_speed_val, spline_type):
    if spline_type == 't':
        for k in range(len(t_list)):
            plt.plot(t_list[k], cs_val_list[k], color='blue')
            plt.plot(t_list[k], lag_pos_val[k], color='red')
            plt.scatter([t_list[k][0], t_list[k][-1]], [cs_val_list[k][0], cs_val_list[k][-1]], color='green')
            plt.scatter([t_list[k][0], t_list[k][-1]], [lag_pos_val[k][0], lag_pos_val[k][-1]], color='green')
        plt.title('Time - Position Plot')
        plt.xlabel('time')
        plt.ylabel('Position')
        plt.grid()
        plt.show()

        for k in range(len(t_list)):
            plt.plot(t_list[k], dcs_val_list[k], color='blue')
            plt.plot(t_list[k], lag_speed_val[k], color='red')
            plt.scatter([t_list[k][0], t_list[k][-1]], [dcs_val_list[k][0], dcs_val_list[k][-1]], color='green')
            plt.scatter([t_list[k][0], t_list[k][-1]], [lag_speed_val[k][0], lag_speed_val[k][-1]], color='green')
        plt.title('Time - Speed Plot')
        plt.xlabel('time')
        plt.ylabel('speed')
        plt.grid()
        plt.show()
    else:
        print("here")
        plt.figure(figsize=(20, 5))
        for k in range(len(t_list)):
            plt.plot(t_list[k], cs_val_list[k], '-b')
            plt.plot(t_list[k], lag_pos_val[k], '-r')
            plt.scatter([t_list[k][0], t_list[k][-1]], [cs_val_list[k][0], cs_val_list[k][-1]], color='blue')
            plt.scatter([t_list[k][0], t_list[k][-1]], [lag_pos_val[k][0], lag_pos_val[k][-1]], color='red')
        plt.title('Bird Plot')
        plt.xlabel('x - axis')
        plt.ylabel('f(x)')
        plt.xlim(0, 14)
        plt.ylim(-2, 6)
        plt.grid()
        plt.show()


def over_speed(max_speed, t_list, speed_list):
    not_found = True

    while not_found:
        for k in range(len(t_list)):
            index = 0
            for num in speed_list[k]:
                if num > max_speed:
                    print(f"The car exceeds {max_speed} MPH at {t_list[k][index]} s")
                    not_found = False
                    break
                if not not_found:
                    break
                else:
                    index += 1
            if not not_found:
                break


t_input = []
x_input = []
x_dot_input = []

question = input("Tutorial (clamped vs lagrange) (t) or bird shape (un clamped) (b): ").lower()

if question == 't':

    t_input = [0, 3, 5, 8, 13]
    x_input = [0, 225, 383, 623, 993]
    x_dot_input = [75, 77, 80, 74, 72]

    lagrange_poly = calc_lagrange_polynomial(t_input, x_input)

    no_of_segments = len(x_input) - 1
    [coefficient_a, coefficient_b, coefficient_c, coefficient_d] = calc_cs_coefficients(t_input, x_input, x_dot_input,
                                                                                        question)
    y = symbols('y')

    cs_list = []
    dcs_dy_list = []

    for j in range(no_of_segments):
        cs_list.append(coefficient_a[j] + coefficient_b[j] * (y - t_input[j]) + coefficient_c[j] * (y - t_input[j]) ** 2
                       + coefficient_d[j] * (y - t_input[j]) ** 3)
        dcs_dy_list.append(cs_list[j].diff(y))

    y_list = []
    for j in range(no_of_segments):
        y_list.append(np.linspace(t_input[j], t_input[j + 1], 20))

    [cs_pos_list, cs_spd_list, minim_speed, time1, maxim_speed, time2] = calc_pos_speed(y_list, cs_list, dcs_dy_list)

    [lag_pos_list, lag_speed_list] = calc_lagrange_value(lagrange_poly, y_list)

    graph_input = input("Do you want to see the plots of displacement and speed?(y/n): ")
    if graph_input.lower() == 'y':
        plot_graph(y_list, cs_pos_list, cs_spd_list, lag_pos_list, lag_speed_list, question)

    print('Calculations are done according to cubic spline curve.')

    print(f"Minimum speed was {minim_speed} MPH at {time1} s, and maximum speed was {maxim_speed} MPH at {time2} s.")

    over_speed_input = input("Do you want to check for over speed?(y/n): ")
    if over_speed_input.lower() == 'y':
        perm_speed = float(input("Enter the maximum permissible speed (MPH): "))
        over_speed(perm_speed, y_list, cs_spd_list)

    pos_speed_input = input("Do you want to check for position and speed at some time?(y/n): ")
    if pos_speed_input == 'y':
        input_time = float(input("Enter the time (s) at which you want to check the speed and position of vehicle: "))
        calc_pos_speed_at_time(y_list, cs_list, dcs_dy_list, input_time)
        calc_lagrange_val_at_time(lagrange_poly, input_time)

    print("Thank You")

elif question == 'b':
    t_input = [0.9, 1.3, 1.9, 2.1, 2.6, 3.0, 3.9, 4.4, 4.7, 5.0, 6.0, 7.0, 8.0, 9.2, 10.5, 11.3, 11.6, 12.0, 12.6, 13.0,
               13.3]
    x_input = [1.3, 1.5, 1.85, 2.1, 2.6, 2.7, 2.4, 2.15, 2.05, 2.1, 2.25, 2.3, 2.25, 1.95, 1.4, 0.9, 0.7, 0.6, 0.5, 0.4,
               0.25]
    x_dot_input = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    lagrange_poly = calc_lagrange_polynomial(t_input, x_input)

    no_of_segments = len(x_input) - 1
    [coefficient_a, coefficient_b, coefficient_c, coefficient_d] = calc_cs_coefficients(t_input, x_input, x_dot_input,
                                                                                        question)
    y = symbols('y')

    cs_list = []
    dcs_dy_list = []

    for j in range(no_of_segments):
        cs_list.append(coefficient_a[j] + coefficient_b[j] * (y - t_input[j]) + coefficient_c[j] * (y - t_input[j]) ** 2
                       + coefficient_d[j] * (y - t_input[j]) ** 3)
        dcs_dy_list.append(cs_list[j].diff(y))

    y_list = []
    for j in range(no_of_segments):
        y_list.append(np.linspace(t_input[j], t_input[j + 1], 10))

    [pos_list, spd_list, minim_speed, time1, maxim_speed, time2] = calc_pos_speed(y_list, cs_list, dcs_dy_list)

    [lag_pos_list, lag_speed_list] = calc_lagrange_value(lagrange_poly, y_list)

    plot_graph(y_list, pos_list, spd_list, lag_pos_list, lag_speed_list, question)
    print("Thank You")
