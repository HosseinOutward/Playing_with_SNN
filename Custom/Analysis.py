if __name__ == "__main__":
    from Ex_1.analysis import plot_mv_ms
    from Ex_2.analysis import *
    from Ex_3.Learning_Alg import *

    dt=0.03125;runtime=100
    time_steps=int(runtime//dt)
    model=CustomModel(n_type=IF, excit_count=5, inhib_count=0, J=1,
        delay_range=(0,0), delay_seg=1, stdp_eng=None,
        n_config="dt="+str(dt)+", R=10, tau=8, theta=-45, U_rest=-65, U_reset=-65, U_spike=5, "
        "weight_sens=1,ref_period=0, ref_time=0, theta_rh=-45, delta_t=2, a=0.01, b=500, tau_k=100")

    u_history=[]; i_history=[]; w_his=[]; train_delay=50
    freq = lambda x,f: 1 if (x*dt-f*1)%train_delay==0 else 0
    curr_func1 = lambda x: 6000
    curr_func2 = lambda x: 0
    for t in range(time_steps):
        u, cur = model.simulate_network_one_step(curr_func1(t),curr_func2(t))
        w_his.append([o_n.pre_syn[0].w for o_n in model.output_neurons])
        u_history.append(u)
        i_history.append(cur)

    l=len(model.input_neurons[0].t_fired)
    print(model.a)
    print(l)
    print(65+model.input_neurons[0].U)

    print(10/1000 * curr_func1(1)*dt / 8 * (runtime/dt-l)/20)

    min_t = runtime-25
    # model.draw_graph()
    plot_mv_ms(array(u_history), arange(0,runtime, dt), top=-15, bottom=-66, max_x=min_t)
    plot_mv_ms(array(i_history), arange(0,runtime, dt), max_x=0)
    plot_raster(*generate_spike_data(model, runtime, dt), runtime, dt, min_t=0)
    plot_raster(*generate_spike_data(model, runtime, dt), runtime, dt, min_t=min_t)
    # w_his=np.array(w_his)
    # plot_weight_spike(dt,runtime,model.pre_neurons[0], 0, w_his[:,0], max_x=0)
    # plot_weight_spike(dt,runtime,model.pre_neurons[0], 1, w_his[:,1], max_x=0)
