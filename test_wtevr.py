from dropconnect_tensorflow import ldpc_gen as ldpc

n = 64
wc = 4
q = 1
R_code = 0.5
is_code_loaded = 0
infobit, nr, H, H_sparse, H_vals, rate, _, _ = ldpc.generate_ldpc_code(n, wc, q, R_code, is_code_loaded)
print("The number of infomation bit: %d \n The nmumber of rows: %d \n The true rate of the code: %f" %(infobit, nr, rate))
print(H)