def print_args(args):
    print("\n\033[1m" + "Basic Config" + "\033[0m")
    print(f'  {"Task Name:":<20}{args.task_name:<20}{"Is Training:":<20}{args.is_training:<20}')
    print(f'  {"Random Seed:":<20}{args.random_seed:<20}{"Channel Independence:":<20}{args.ch_ind:<20}')
    print(f'  {"Model ID:":<20}{args.model_id:<20}{"Model:":<20}{args.model:<20}')
    print(f'  {"Partial Train:":<20}{args.partial_train:<20}')

    print("\n\033[1m" + "Data Loader" + "\033[0m")
    print(f'  {"Data:":<20}{args.data:<20}{"Root Path:":<20}{args.root_path:<20}')
    print(f'  {"Data Path:":<20}{args.data_path:<20}{"Features:":<20}{args.features:<20}')
    print(f'  {"Target:":<20}{args.target:<20}{"Freq:":<20}{args.freq:<20}')
    print(f'  {"Checkpoints:":<20}{args.checkpoints:<20}')

    if args.task_name in ['long_term_forecast', 'short_term_forecast']:
        print("\n\033[1m" + "Forecasting Task" + "\033[0m")
        print(f'  {"Seq Len:":<20}{args.seq_len:<20}{"Pred Len:":<20}{args.pred_len:<20}')
        print(f'  {"Inverse:":<20}{args.inverse:<20}{"Label Len:":<20}{args.label_len:<20}')
        # print(f'  {"Pred Len:":<20}{args.pred_len:<20}{"Seasonal Patterns:":<20}{args.seasonal_patterns:<20}')

    if args.task_name == 'imputation':
        print("\n\033[1m" + "Imputation Task" + "\033[0m")
        print(f'  {"Mask Rate:":<20}{args.mask_rate:<20}')

    if args.task_name == 'anomaly_detection':
        print("\n\033[1m" + "Anomaly Detection Task" + "\033[0m")
        print(f'  {"Anomaly Ratio:":<20}{args.anomaly_ratio:<20}')

    print("\n\033[1m" + "General Model Parameters" + "\033[0m")
    # print(f'  {"Top k:":<20}{args.top_k:<20}{"Num Kernels:":<20}{args.num_kernels:<20}')
    print(f'  {"Use Norm:":<20}{args.revin:<20}{"Affine:":<20}{args.affine:<20}')
    print(f'  {"Enc In:":<20}{args.enc_in:<20}{"Dec In:":<20}{args.dec_in:<20}')
    print(f'  {"e layers:":<20}{args.e_layers:<20}{"d layers:":<20}{args.d_layers:<20}')
    print(f'  {"d model:":<20}{args.d_model:<20}{"d FF:":<20}{args.d_ff:<20}')
    print(f'  {"Dropout:":<20}{args.dropout:<20}{"Activation:":<20}{args.activation:<20}')
    print(f'  {"C Out:":<20}{args.c_out:<20}{"n heads:":<20}{args.n_heads:<20}')
    print(f'  {"Moving Avg:":<20}{args.moving_avg:<20}{"Factor:":<20}{args.factor:<20}')
    print(f'  {"Embed:":<20}{args.embed:<20}{"Distil:":<20}{args.distil:<20}')

    if args.model=='SOFTS':
        print("\n\033[1m" + "SOFTS Parameters" + "\033[0m")
        print(f'  {"d_core:":<20}{args.d_core:<20}')
    elif args.model=='TimeMixer':
        print("\n\033[1m" + "TimeMixer Parameters" + "\033[0m")
        print(f'  {"down_sampling_layers:":<20}{args.down_sampling_layers:<20}'
              f'{"down_sampling_window:":<20}{args.down_sampling_window:<20}')
        print(f'  {"down_sampling_method:":<20}{args.down_sampling_method:<20}'
              f'{"decomp_method:":<20}{args.decomp_method:<20}')
        print(f'  {"use_future_temporal_feature:":<20}{args.use_future_temporal_feature:<20}')
    elif 'former' in args.model:
        print("\n\033[1m" + "Transformer Parameters" + "\033[0m")
        print(f'  {"Output Attention:":<20}{args.output_attention:<20}')
    elif 'Mamba' in args.model:
        print("\n\033[1m" + "Mamba Parameters" + "\033[0m")
        print(f'  {"d_state:":<20}{args.d_state:<20}{"dconv:":<20}{args.dconv:<20}')
        print(f'  {"e_fact:":<20}{args.e_fact:<20}{"is_flip:":<20}{args.is_flip:<20}')

    print("\n\033[1m" + "Run Parameters" + "\033[0m")
    print(f'  {"Num Workers:":<20}{args.num_workers:<20}{"Itr:":<20}{args.itr:<20}')
    print(f'  {"Batch Size:":<20}{args.batch_size:<20}{"Learning Rate:":<20}{args.learning_rate:<20}')
    print(f'  {"Train Epochs:":<20}{args.train_epochs:<20}{"Patience:":<20}{args.patience:<20}')
    # print(f'  {"Des:":<20}{args.des:<20}{"Loss:":<20}{args.loss:<20}')
    print(f'  {"Lradj:":<20}{args.lradj:<20}{"Pct Start:":<20}{args.pct_start:<20}')
    print(f'  {"Use Amp:":<20}{args.use_amp:<20}{"Use Wandb:":<20}{args.use_wandb:<20}')
    print(f'  {"Basic Loss:":<20}{args.alpha}{" * MAE + "}{(1-args.alpha)}{" * MSE"}')
    print(f'  {"Des:":<20}{args.des:<20}')

    print("\n\033[1m" + "GPU" + "\033[0m")
    print(f'  {"Use GPU:":<20}{args.use_gpu:<20}{"GPU:":<20}{args.gpu:<20}')
    print(f'  {"Use Multi GPU:":<20}{args.use_multi_gpu:<20}{"Devices:":<20}{args.devices:<20}')

    if args.is_cluster:
        print("\n\033[1m" + "Cluster Parameters" + "\033[0m")
        print(f'  {"Num of Clusters:":<20}{args.n_clusters:<20}{"Corr Threshold:":<20}{args.corr_threshold:<20}')

    if args.is_reindex:
        print("\n\033[1m" + "Data Augmentation Parameters" + "\033[0m")
        print(f'  {"Is Reindex:":<20}{args.is_reindex:<20}{"Adj K:":<20}{args.adj_k:<20}')

    # print("\033[1m" + "De-stationary Projector Params" + "\033[0m")
    # p_hidden_dims_str = ', '.join(map(str, args.p_hidden_dims))
    # print(f'  {"P Hidden Dims:":<20}{p_hidden_dims_str:<20}{"P Hidden Layers:":<20}{args.p_hidden_layers:<20}')
    print()
