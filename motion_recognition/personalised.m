%% Personalised calibration + real‑time comparison
clear; clc;

% Set Python environment
pyenv('Version', 'E:\MachineLearning\ml_env\Scripts\python.exe');
insert(py.sys.path, 0, 'E:\MachineLearning\IndividualProject');
py.importlib.import_module('motion_recognition.inference.realtime_predictor');

% Connect phone
m = mobiledev;
m.AccelerationSensorEnabled = 1;
m.Logging = 0;
discardlogs(m);
disp('Phone connected. Starting calibration...');

% Mode selection
fprintf('\nSelect calibration mode:\n');
fprintf('  1: Check action existence (keep valid windows)\n');
fprintf('  2: Direct collection (keep all windows)\n');
fprintf('  3: Use existing personalised model (skip calibration)\n');
mode = input('Enter mode (1,2,3): ');
while ~ismember(mode, [1,2,3])
    mode = input('Invalid. Enter 1,2,3: ');
end
fprintf('Mode %d selected.\n', mode);

% If mode 3, check model exists
if mode == 3
    custom_path = fullfile('E:\MachineLearning\IndividualProject\motion_recognition\output\models\calibrated', 'current_user_big_classifier.h5');
    if ~exist(custom_path, 'file')
        error('Personalised model not found. Run mode 1 or 2 first.');
    end
    fprintf('Found existing model. Skipping calibration.\n');
else
    % Mode 1 or 2: collect 5 samples per action
    actions = {'right_down', 'left_down', 'circle_clock', 'circle_anticlock'};
    samples_per_action = 1;
    for i = 1:length(actions)
        act = actions{i};
        collected = 0;
        while collected < samples_per_action
            fprintf('\nPerform action: %s (%d/%d) then press Enter', act, collected+1, samples_per_action);
            input('');
            m.Logging = 0; discardlogs(m); pause(0.1);
            m.Logging = 1; pause(1.5); m.Logging = 0;
            [accel, timestamp] = accellog(m);
            if length(timestamp) < 5
                fprintf('Insufficient data, retry.\n');
                continue;
            end
            n = min(100, length(timestamp));
            data = [timestamp(end-n+1:end), accel(end-n+1:end, :)];
            result = py.motion_recognition.inference.realtime_predictor.collect_calibration_sample(act, data, int32(mode));
            if result{'success'}
                fprintf('✓ %s sample %d collected\n', act, collected+1);
                collected = collected + 1;
            else
                msg = char(result{'message'});
                fprintf('✗ Failed: %s, retry.\n', msg);
            end
        end
    end
    % Fine‑tune
    fprintf('\nAll samples collected. Fine‑tuning...\n');
    result = py.motion_recognition.inference.realtime_predictor.finish_calibration('current_user', 5);
    try
        if strcmp(char(result{'status'}), 'success')
            fprintf('✓ Calibration done. Model saved: %s\n', char(result{'model_path'}));
        else
            fprintf('✗ Calibration failed: %s\n', char(result{'message'}));
        end
    catch
        fprintf('✗ Calibration returned unexpected format.\n');
    end
end

% Real‑time comparison
fprintf('\nEntering real‑time comparison mode. Press Ctrl+C to stop.\n');
fprintf('%-8s %-20s %-10s %-20s %-10s\n', 'Time(s)', 'Original', 'Conf', 'Personalised', 'Conf');
fprintf('----------------------------------------------------------------\n');

m.Logging = 1;
lastPrint = tic;
minInterval = 0.5;
offset = [];

last_orig = 'waiting'; orig_conf = 0;
last_cust = 'waiting'; cust_conf = 0;

try
    while m.AccelerationSensorEnabled == 1
        [accel, ts] = accellog(m);
        if ~isempty(ts)
            t = ts(end);
            a = accel(end,:);
            sensor = [t, a(1), a(2), a(3)];
            if isempty(offset)
                offset = t;
            end
            rel = t - offset;
            try
                r_orig = py.motion_recognition.inference.realtime_predictor.predict_original(sensor);
                last_orig = char(r_orig{'action'});
                orig_conf = double(r_orig{'confidence'});
            catch
            end
            try
                r_cust = py.motion_recognition.inference.realtime_predictor.predict_custom(sensor);
                last_cust = char(r_cust{'action'});
                cust_conf = double(r_cust{'confidence'});
            catch
            end
        end
        if toc(lastPrint) >= minInterval
            lastPrint = tic;
            if ~isempty(ts)
                fprintf('%-8.1f %-20s %-10.3f %-20s %-10.3f\n', rel, last_orig, orig_conf, last_cust, cust_conf);
            end
        end
        pause(0.02);
    end
catch ME
    if ~strcmp(ME.identifier, 'MATLAB:class:InvalidHandle')
        disp(ME.message);
    end
end
m.Logging = 0;
delete(m);
disp('Test ended.');