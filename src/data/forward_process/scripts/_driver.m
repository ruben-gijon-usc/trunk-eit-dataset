run('/home/ruben/Documentos/trunk-eit-dataset/src/data/forward_process/scripts/startup_eidors.m');
addpath('/home/ruben/Documentos/trunk-eit-dataset/src/data/forward_process/scripts');
startup_eidors();
run_forward(
    16,
    'opposite',
    '/tmp/tmp5g1ohfcm/grid.mat',
    -1.0, 1.0,
    -1.0, 1.0,
    1.565712225547997,
    '/tmp/tmp5g1ohfcm'
);