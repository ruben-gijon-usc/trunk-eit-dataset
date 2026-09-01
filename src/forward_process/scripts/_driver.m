run('/home/ruben/Documentos/trunk-eit-dataset/src/forward_process/scripts/startup_eidors.m');
addpath('/home/ruben/Documentos/trunk-eit-dataset/src/forward_process/scripts');
startup_eidors();
run_forward(
    16,
    '/tmp/tmpcloooq_e/grid.txt',
    -0.2949127982472068, 0.2949127982472068,
    -0.2949127982472068, 0.2949127982472068,
    0.01281065400287076,
    '/tmp/tmpcloooq_e'
);