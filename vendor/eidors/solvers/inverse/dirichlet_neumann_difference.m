function del_L = dirichlet_neumann_difference(fmdl,data1,data2);
% TODO:
%  - consider to implement Harach https://iopscience.iop.org/article/10.1088/0266-5611/31/11/115008/pdf
%  - maybe other approaches 
%
% (C) 2023 Joeran Rixen. Licensed under GPL version 2 or 3

% TODO: allow choosing the approach, only have one for now
del_L =  d2n_from_full_data(fmdl,data1,data2);

end

function del_L = d2n_from_full_data(fmdl,data1,data2);
    %% Calculations regarding the Dirichlet to Neumann map
    stim_pattern_mat = create_stim_pattern_matrix(fmdl);
    stim_pattern_mat = stim_pattern_mat./sqrt(sum(stim_pattern_mat.^2,1));

    n_elec = num_elecs(fmdl);

    data1 = calc_difference_data(0,data1,fmdl);
    data2 = calc_difference_data(0,data2,fmdl);

    % reshaping voltages for easy BN matrix assembly
    data1 = reshape(data1, [n_elec, n_elec-1]);
    data2 = reshape(data2, [n_elec, n_elec-1]);
    
    data1 = make_voltage_mean_free(data1);
    data2 = make_voltage_mean_free(data2);
    
    data1 = data1./sqrt(sum(stim_pattern_mat.^2,1));
    data2 = data2./sqrt(sum(stim_pattern_mat.^2,1));
    
    %either calcualte or give, as this is rather imporant for scaling of
    %the output of the d-bar algorithm.
    elec_area = pi*0.25^2;

    NtoD1 = (1/elec_area)*(full(stim_pattern_mat)'*data1);
    NtoD2 = (1/elec_area)*(full(stim_pattern_mat)'*data2);
    
    DN1 = inv(NtoD1);
    DN2 = inv(NtoD2);

    del_L = (DN1 - DN2);
end
    

function stim_pattern_mat = create_stim_pattern_matrix(fmdl)
    % converts the stimpattern in the model into matrix form for ease of
    % computation
    n_elec = length(fmdl.electrode);
    stim_pattern_mat = zeros(n_elec, n_elec-1);
    
    for i = 1:n_elec-1
        stim_pattern_mat(:, i) = full(fmdl.stimulation(i).stim_pattern);
    end
end

function data_mat_free = make_voltage_mean_free(data_mat)
    % makes each stimulation voltage vector mean free
    U_mean = mean(data_mat, 1);
    data_mat_free = data_mat - U_mean;
end

