%% 实时动作识别（优化：只处理最新数据点，控制打印频率）
clear; clc;

% 1. 设置Python环境
disp('设置Python环境...');
pyenv('Version', 'E:\MachineLearning\ml_env\Scripts\python.exe');
insert(py.sys.path, 0, 'E:\MachineLearning\IndividualProject');
py.importlib.import_module('motion_recognition.inference.realtime_predictor');
disp('✓ Python环境设置成功');

% 2. 连接手机传感器
disp('连接手机传感器...');
try
    m = mobiledev;
    m.AccelerationSensorEnabled = 1;
    m.SampleRate = 10;          % 10Hz采样率
    m.Logging = 0;               % 先不启动
    disp('✓ 手机连接成功');
catch ME
    error('无法连接手机: %s', ME.message);
end

% 3. 初始化变量
fprintf('\n开始实时识别，按 Ctrl+C 停止...\n');
lastProcessedTime = -inf;        % 上次处理的时间戳
lastPrintTime = tic;             % 打印计时器
lastResult = struct('action', 'no_action', 'confidence', 0, 'action_present', false);

% 4. 启动数据记录
m.Logging = 1;

% 5. 主循环
try
    while m.AccelerationSensorEnabled == 1
        pause(0.05);             % 50ms轮询一次
        
        % 获取最新一条数据
        [accel, timestamp] = accellog(m);
        if isempty(timestamp)
            continue;
        end
        
        latestTime = timestamp(end);
        latestAccel = accel(end, :);
        
        % 如果有新数据，则处理
        if latestTime > lastProcessedTime
            sensorData = [latestTime, latestAccel(1), latestAccel(2), latestAccel(3)];
            result = py.motion_recognition.inference.realtime_predictor.predict(sensorData);
            if isa(result, 'py.dict')
                lastResult.action = char(result{'action'});
                lastResult.confidence = double(result{'confidence'});
                lastResult.action_present = logical(result{'action_present'});
            end
            lastProcessedTime = latestTime;
        end
        
        % 每0.5秒打印一次最新结果
        if toc(lastPrintTime) >= 0.3
            lastPrintTime = tic;
            fprintf('[t=%.1fs] %s (%.2f) 有动作=%d\n', ...
                latestTime, lastResult.action, lastResult.confidence, lastResult.action_present);
        end
    end
catch ME
    % 用户中断或其他错误处理
    if strcmp(ME.identifier, 'MATLAB:class:InvalidHandle') || ...
       strcmp(ME.identifier, 'MATLAB:mir_warning:maybe_infinite_loop_detected')
        disp('用户中断，正在清理...');
    else
        disp('程序出错:');
        disp(ME.message);
    end
end

% 6. 清理
m.Logging = 0;
delete(m);
disp('测试结束');