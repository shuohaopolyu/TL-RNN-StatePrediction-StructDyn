from systems import BaseIsolatedStructure
import numpy as np
import pickle
import matplotlib.pyplot as plt


def _read_smc_file(filename):
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


def compute_response(num=1):
    acc_file_root_name = "./excitations/SMSIM/m7.0r10.0_00"

    acc_file_list = [
        acc_file_root_name + format(i, "03") + ".smc" for i in range(1, num + 1)
    ]

    for acc_file_i in acc_file_list:
        time, acc = _read_smc_file(acc_file_i)
        time = time[1000:6000]
        acc = acc[1000:6000]
        time = time - time[0]
        mass_vec = 3e5 * np.ones(12)
        stiff_vec = 9e6 * np.ones(12)
        damp_vec = 3e5 * np.ones(12)
        parametric_bists = BaseIsolatedStructure(
            mass_super_vec=mass_vec,
            stiff_super_vec=stiff_vec,
            damp_super_vec=damp_vec,
            isolator_params={
                "m_b": 6e5,
                "c_b": 5e5,
                "k_b": 3e6,
                "q": 4e-1,
                "A": 0,
                "beta": 0,
                "gamma": 0,
                "n": 1,
                "z_0": 0,
                "F_y": 0,
                "alpha": 1,
            },
            x_0=np.zeros(13),
            x_dot_0=np.zeros(13),
            t=time,
            acc_g=acc,
        )

        disp, velo, acc, z = parametric_bists.run()

        solution = {
            "acc_g": parametric_bists.acc_g,
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
        # _ = bis.print_damping_ratio(10)
        _ = parametric_bists.print_natural_frequency(10)
    return solution


def analytical_validation():
    time = np.linspace(0, 10, 10000)
    acc = np.sin(2 * np.pi * 1 * time)
    mass_vec = 2 * np.ones(2)
    stiff_vec = 10 * np.ones(2)
    damp_vec = 0.1 * np.ones(2)
    parametric_bists = BaseIsolatedStructure(
        mass_super_vec=mass_vec,
        stiff_super_vec=stiff_vec,
        damp_super_vec=damp_vec,
        isolator_params={
            "m_b": 1,
            "c_b": 0.1,
            "k_b": 10,
            "q": 4e-3,
            "A": 0,
            "beta": 0,
            "gamma": 0,
            "n": 1,
            "z_0": 0,
            "F_y": 0,
            "alpha": 1,
        },
        x_0=np.zeros(3),
        x_dot_0=np.zeros(3),
        t=time,
        acc_g=acc,
    )

    disp, velo, acc, z = parametric_bists.run()
    _ = parametric_bists.print_natural_frequency(3)

    solution = {
        "acc_g": parametric_bists.acc_g,
        "time": time,
        "disp": disp,
        "velo": velo,
        "acc": acc,
        "z": z,
    }
    return solution


def plot_response():
    # free to modify
    with open("./dataset/base_isolated_structure/solution002.pkl", "rb") as f:
        solution = pickle.load(f)
    time = solution["time"]
    acc_g = solution["acc_g"]
    disp = solution["disp"]
    velo = solution["velo"]
    acc = solution["acc"]
    z = solution["z"]
    plt.figure()
    plt.plot(time, acc_g)
    plt.xlabel("Time (s)")
    plt.ylabel("Ground acceleration (m/s^2)")
    plt.figure()
    plt.plot(time, disp[0, :])
    plt.xlabel("Time (s)")
    plt.ylabel("Base displacement (m)")
    plt.figure()
    plt.plot(time, velo[0, :])
    plt.xlabel("Time (s)")
    plt.ylabel("Base velocity (m/s)")
    plt.figure()
    plt.plot(time, z[0, :])
    plt.xlabel("Time (s)")
    plt.ylabel("Isolator displacement (m)")
    plt.figure()
    plt.plot(time, acc[0, :])
    plt.xlabel("Time (s)")
    plt.ylabel("Base acceleration (m/s^2)")
    plt.show()

