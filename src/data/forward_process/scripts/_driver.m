run('/home/ruben/Documentos/trunk-eit-dataset/src/data/forward_process/scripts/startup_eidors.m');
addpath('/home/ruben/Documentos/trunk-eit-dataset/src/data/forward_process/scripts');
startup_eidors();
run_forward(
    16,
    'opposite',
    '/tmp/tmpe3zou1my/grid.mat',
    -1.0, 1.0,
    -1.0, 1.0,
    0.8672163877271393,
    '/tmp/tmpe3zou1my'
);