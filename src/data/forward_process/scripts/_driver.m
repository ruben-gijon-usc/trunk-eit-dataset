run('/home/ruben/Documentos/trunk-eit-dataset/src/data/forward_process/scripts/startup_eidors.m');
addpath('/home/ruben/Documentos/trunk-eit-dataset/src/data/forward_process/scripts');
startup_eidors();
run_forward(
    16,
    'opposite',
    '/tmp/tmpaazj1mvx/grid.mat',
    -1.0, 1.0,
    -1.0, 1.0,
    1.487055043683915,
    '/tmp/tmpaazj1mvx'
);