%% Semi‑automatic data collection and annotation
clear; clc;

% User configuration
dataRoot = 'E:\MachineLearning\IndividualProject\motion_data\training_data\new_data';
actions = {'right_down', 'left_down', 'circle_clock', 'circle_anticlock'};
abbr = {'RD', 'LD', 'CC', 'CA'};

startNumber = input('Enter starting number (two digits, e.g. 01): ', 's');
while isempty(startNumber) || ~isnumeric(str2double(startNumber))
    startNumber = input('Invalid, enter two digits: ', 's');
end
startNum = str2double(startNumber);

startActionIdx = input('Starting action index (1=RD,2=LD,3=CC,4=CA, default=1): ');
if isempty(startActionIdx) || startActionIdx<1 || startActionIdx>4
    startActionIdx = 1;
end

% Create directories
if ~exist(dataRoot,'dir'), mkdir(dataRoot); end
motionDir = fullfile(dataRoot,'motion');
annoDir = fullfile(dataRoot,'annotation');
if ~exist(motionDir,'dir'), mkdir(motionDir); end
if ~exist(annoDir,'dir'), mkdir(annoDir); end

% Connect phone
disp('Connecting phone...');
try
    m = mobiledev;
    m.AccelerationSensorEnabled = 1;
    m.SampleRate = 10;
    m.Logging = 0;
    disp('✓ Connected');
catch ME
    error('Failed to connect: %s', ME.message);
end

% Main loop
currentNumber = startNum;
fprintf('\nStarting collection from number %02d, action %d (%s)\n', currentNumber, startActionIdx, actions{startActionIdx});
fprintf('Sequence: %s\n', strjoin(actions,' -> '));
fprintf('After each capture, you can accept (y) or redo (n).\n');
fprintf('Press Ctrl+C to stop.\n');

try
    while true
        for actIdx = startActionIdx:length(actions)
            act = actions{actIdx};
            abb = abbr{actIdx};
            filename = sprintf('m%s%02d', abb, currentNumber);
            motionFile = fullfile(motionDir, [filename, '.csv']);
            annoFile = fullfile(annoDir, ['a', filename(2:end), '.csv']);
            
            accepted = false;
            while ~accepted
                fprintf('\n===== Collect: %s (%s) #%02d =====\n', act, abb, currentNumber);
                input('Press Enter to start recording...');
                m.Logging = 0; discardlogs(m); pause(0.1);
                m.Logging = 1; pause(3); m.Logging = 0;
                [accel, ts] = accellog(m);
                if length(ts) < 5
                    fprintf('Too few samples, retry.\n');
                    continue;
                end
                rows = 1:length(ts);
                figure('Name', sprintf('%s (%s)', act, abb));
                plot(rows, accel(:,1), 'r-', rows, accel(:,2), 'g-', rows, accel(:,3), 'b-', 'LineWidth',1.5);
                xlabel('Row index'); ylabel('Acceleration'); title(sprintf('%s (%s) - Accept?', act, abb));
                legend('X','Y','Z'); grid on;
                accept = input('\nAccept this recording? (y/n, default y): ', 's');
                if isempty(accept) || lower(accept)=='y'
                    nRows = length(ts);
                    fprintf('\nData has %d rows. Enter start and end row indices.\n', nRows);
                    startRow = []; endRow = [];
                    while isempty(startRow) || startRow<1 || startRow>nRows
                        startRow = input('Start row: ');
                        if isempty(startRow) || ~isnumeric(startRow) || startRow<1 || startRow>nRows
                            fprintf('Invalid, enter 1..%d\n', nRows);
                            startRow = [];
                        end
                    end
                    while isempty(endRow) || endRow<startRow || endRow>nRows
                        endRow = input('End row: ');
                        if isempty(endRow) || ~isnumeric(endRow) || endRow<startRow || endRow>nRows
                            fprintf('Invalid, enter %d..%d\n', startRow, nRows);
                            endRow = [];
                        end
                    end
                    close(gcf);
                    startTime = ts(startRow);
                    endTime = ts(endRow);
                    dataTable = table(ts, accel(:,1), accel(:,2), accel(:,3), ...
                        'VariableNames', {'timestamp','accel_x','accel_y','accel_z'});
                    writetable(dataTable, motionFile);
                    annoTable = table(startTime, endTime, {act}, 1.0, ...
                        'VariableNames', {'start_time','end_time','action_label','confidence'});
                    writetable(annoTable, annoFile);
                    fprintf('✓ Saved %s and %s\n', motionFile, annoFile);
                    accepted = true;
                else
                    close(gcf);
                    fprintf('Redoing current action.\n');
                end
            end
        end
        % Round completed
        startActionIdx = 1;
        fprintf('\n===== Round %d completed =====\n', currentNumber - startNum + 1);
        currentNumber = currentNumber + 1;
        fprintf('Next round starts from number %02d.\n', currentNumber);
        cont = input('Continue next round? (y/n, default y): ', 's');
        if ~isempty(cont) && lower(cont)=='n'
            break;
        end
    end
catch ME
    if ~strcmp(ME.identifier, 'MATLAB:class:InvalidHandle')
        fprintf('Error: %s\n', ME.message);
    end
end
m.Logging = 0;
delete(m);
fprintf('Program ended. Next time start from number %02d, action %d.\n', currentNumber, startActionIdx);