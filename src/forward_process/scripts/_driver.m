run('/home/ruben/Documentos/trunk-eit-dataset/src/forward_process/scripts/startup_eidors.m');
addpath('/home/ruben/Documentos/trunk-eit-dataset/src/forward_process/scripts');
startup_eidors();
run_forward(
    16,
    '/tmp/tmpg9d9axg3/grid.txt',
    -0.2719970471478358, 0.2719970471478358,
    -0.2719970471478358, 0.2719970471478358,
    0.007506096030166212,
    '/tmp/tmpg9d9axg3'
);