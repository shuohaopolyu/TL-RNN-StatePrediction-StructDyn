from systems import ShearTypeStructure
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


def compute_response(num=2, method="DOP853"):
    acc_file_root_name = "./excitations/SMSIM/m7.0r10.0_00"

    acc_file_list = [
        acc_file_root_name + format(i, "03") + ".smc" for i in range(1, num + 1)
    ]

    for acc_file_i in acc_file_list:
        time, acc = _read_smc_file(acc_file_i)
        time = time[1000:6000]
        acc = acc[1000:6000]
        time = time - time[0]
        mass_vec = 5e5 * np.ones(13)
        # mass_vec[0] = 1e6
        stiff_vec = 1e7 * np.ones(13)
        # stiff_vec[0] = 1.2e8
        damp_vec = 5e5 * np.ones(13)
        # damp_vec[0] = 2e6
        sts = ShearTypeStructure(
            mass_vec=mass_vec,
            stiff_vec=stiff_vec,
            damp_vec=damp_vec,
            t=time,
            acc_g=acc,
        )

        acc, velo, disp = sts.run(method)

        solution = {
            "acc_g": sts.acc_g,
            "time": time,
            "disp": disp,
            "velo": velo,
            "acc": acc,
        }

        return solution

        # file_name = (
        #     "./dataset/shear_type_structure/solution"
        #     + format(acc_file_list.index(acc_file_i) + 1, "03")
        #     + ".pkl"
        # )
        # with open(file_name, "wb") as f:
        #     pickle.dump(solution, f)
        # print("File " + file_name + " saved.")


def analytical_validation(method="DOP853"):
    time = np.linspace(0, 10, 10000)
    acc = np.sin(2 * np.pi * 4 * time)
    mass_vec = 1 * np.ones(3)
    mass_vec[0] = 1
    stiff_vec = 1 * np.ones(3)
    stiff_vec[0] = 1
    damp_vec = 1 * np.ones(3)
    damp_vec[0] = 1
    sts = ShearTypeStructure(
        mass_vec=mass_vec,
        stiff_vec=stiff_vec,
        damp_vec=damp_vec,
        t=time,
        acc_g=acc,
    )

    acc, velo, disp = sts.run(method)

    solution = {
        "acc_g": sts.acc_g,
        "time": time,
        "disp": disp,
        "velo": velo,
        "acc": acc,
    }
    return solution


def plot_response():
    # free to modify
    with open("./dataset/shear_type_structure/solution002.pkl", "rb") as f:
        solution = pickle.load(f)
    time = solution["time"]
    acc_g = solution["acc_g"]
    disp = solution["disp"]
    velo = solution["velo"]
    acc = solution["acc"]
    plt.figure()
    plt.plot(time, acc_g)
    plt.xlabel("Time (s)")
    plt.ylabel("Ground acceleration (m/s^2)")
    plt.figure()
    plt.plot(time, disp[12, :])
    plt.xlabel("Time (s)")
    plt.ylabel("Base displacement (m)")
    plt.figure()
    plt.plot(time, velo[12, :])
    plt.xlabel("Time (s)")
    plt.ylabel("Base velocity (m/s)")
    plt.figure()
    plt.plot(time, acc[12, :])
    plt.xlabel("Time (s)")
    plt.ylabel("Base acceleration (m/s^2)")
    plt.show()
