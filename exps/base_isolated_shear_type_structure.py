from systems import BaseIsolatedStructure
import numpy as np
import pickle
import matplotlib.pyplot as plt


def read_smc_file(filename):
    with open(filename) as f:
        content = f.readlines()
    data = []
    counter = 0
    for x in content:
        if counter > 39:
            s = x
            numbers = [s[i : i + 10] for i in range(0, len(s), 10)]
            number = []
            for i in range(len(numbers) - 1):
                number.append(float(numbers[i]))
            data.append(number)
        counter += 1
    acc = np.array(data)
    time = np.linspace(0, 0.02 * (len(acc[:, 0]) - 1), len(acc[:, 0]))
    return time, acc[:, 0] * 1e-2


def compute_response(num=2):
    acc_file_root_name = "./excitations/SMSIM/m7.0r10.0_00"

    acc_file_list = [
        acc_file_root_name + format(i, "03") + ".smc" for i in range(1, num+1)
    ]

    for acc_file_i in acc_file_list:
        time, acc = read_smc_file(acc_file_i)
        time = time[1000:6000]
        acc = acc[1000:6000]
        time = time - time[0]
        mass_vec = 5e5 * np.ones(12)
        stiff_vec = 1e9 * np.ones(12)
        bis = BaseIsolatedStructure(
            mass_super_vec=mass_vec,
            stiff_super_vec=stiff_vec,
            damp_type="Rayleigh",
            damp_params=(0, 3, 0.03),
            mass_base=1e6,
            isolator_params={
                # 5e5
                "c_b": 0.0,
                "k_b": 1.2e8,
                "q": 4e-2,
                "A": 1,
                "beta": 0.5,
                "gamma": 0.5,
                "n": 2,
                "z_0": 0,
                "F_y": 3.5e6,
                "alpha": 1.0,
            },
            x_0=np.zeros(13),
            x_dot_0=np.zeros(13),
            t=time,
            acc_g=acc,
        )

        disp, velo, acc, z = bis.run()

        solution = {
            "acc_g": bis.acc_g,
            "time": time,
            "disp": disp,
            "velo": velo,
            "acc": acc,
            "z": z,
        }

        file_name = (
            "./dataset/base_isolated_structure/solution"
            + format(acc_file_list.index(acc_file_i) + 1, "03")
            + ".pkl"
        )
        with open(file_name, "wb") as f:
            pickle.dump(solution, f)
        print("File " + file_name + " saved.")

def plot_response():
    # free to modify
    with open('./dataset/base_isolated_structure/solution002.pkl', 'rb') as f:
        solution = pickle.load(f)
    time = solution['time']
    acc_g = solution['acc_g']
    disp = solution['disp']
    velo = solution['velo']
    acc = solution['acc']
    z = solution['z']
    plt.figure()
    plt.plot(time, acc_g)
    plt.xlabel('Time (s)')
    plt.ylabel('Ground acceleration (m/s^2)')
    plt.figure()
    plt.plot(time, disp[0, :])
    plt.xlabel('Time (s)')
    plt.ylabel('Base displacement (m)')
    plt.figure()
    plt.plot(time, velo[0, :])
    plt.xlabel('Time (s)')
    plt.ylabel('Base velocity (m/s)')
    plt.figure()
    plt.plot(time, z[0, :])
    plt.xlabel('Time (s)')
    plt.ylabel('Isolator displacement (m)')
    plt.figure()
    plt.plot(time, acc[0, :])
    plt.xlabel('Time (s)')
    plt.ylabel('Base acceleration (m/s^2)')
    plt.show()
