import numpy as np, os, time
import subprocess

def execute_test_case(type, number_row1, number_col1, number_col2, path_input, path_output):
    # Generate random matrices
    mtx_A = np.random.random(size = (number_row1, number_col1)) * 1e2 # dtype = float64
    mtx_B = np.random.random(size = (number_col1, number_col2)) * 1e2 # dtype = float64

    # Matrix multiplication
    mtx_C = (mtx_A @ mtx_B).flatten() # dtpye = float64

    # Create directories if does not exist
    if not os.path.exists(path_input):
        os.makedirs(path_input)
    if not os.path.exists(path_output):
        os.makedirs(path_output)

    # Store the input matrices
    with open(f"{path_input}/mtx_A.bin", "wb") as fp:
        fp.write(mtx_A.tobytes())
    with open(f"{path_input}/mtx_B.bin", "wb") as fp:
        fp.write(mtx_B.tobytes())
    
    # Compile the students code
    os.system("make")

    output_file = f"out_{number_row1}_{type}.txt"

    print("running for n = ", n, ", type = ", t)
    # Start the timer

    command = f"perf stat -e cache-references,cache-misses,cycles,instructions -d ./main {type} {number_row1} {number_col1} {number_col2} {path_input} {path_output}"

    with open(output_file, "w") as file:
        time_start = time.perf_counter()
        subprocess.run(command, shell=True, stdout=file, stderr=file)
        time_duration = time.perf_counter() - time_start

        file.write("\n\nTime Duration:- " + str(time_duration) +  "\n")

    prof_output = "prof_" + output_file
    os.system(f"gprof main gmon.out > {prof_output}")
    os.system("rm gmon.out")

    # Get the students output matrix
    with open(f"{path_output}/mtx_C.bin", "rb") as fp:
        student_result = np.frombuffer(fp.read(), dtype=mtx_C.dtype)
    
    # Check if the result matrix dimensions match
    if mtx_C.shape != student_result.shape:
        print("The result matrix shape didn't match")
        return False, time_duration
    
    # Check if the student's result is close to the numpy's result within a tolerance
    result = np.allclose(mtx_C, student_result, rtol=1e-10, atol=1e-12)

    return result, time_duration

if __name__ == "__main__":

    l = np.array([1000, 2000, 3000, 4000, 5000])
    # l = np.array([2000])

    types = [0, 1, 2, 3, 4, 5]
    # types = [1]

    type_labels = ["IJK", "IKJ", "JIK", "JKI", "KIJ", "KJI"]

    time_dict = {t: [] for t in types}
    l_p = []

    for n in l:
        for t in types:
            print("running for n = ", n, ", type = ", t)
            result, time_duration = execute_test_case(t, n, n, n, "./input_path/", "./output_path/")
            time_dict[t].append(time_duration)


            if result:
                print("Test Case passed")
            else:
                print("Test Case failed")
            print("\n")

        l_p.append(n)
        # plt.figure(figsize=(10, 6))
        # for t in types:
        #     plt.plot(l_p, time_dict[t], label=f'Type {type_labels[t]}', marker='o')

        # with open("time.json", "w") as json_file:
        #     json.dump(time_dict, json_file, indent=4) 

        # plt.title('Execution Time for Different Loop Permutations')
        # plt.xlabel('Size of Matrix(n)')
        # plt.ylabel('Time Taken (s)')
        # plt.legend()
        # plt.grid()
        # plt.savefig(f"plot{n}.png")