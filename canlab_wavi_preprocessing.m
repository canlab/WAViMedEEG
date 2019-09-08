% The following code is intended for CANlab / WAVi back pain study data
% you should cd into the directory containing files of interest

%load filenames of all 4 CANlab tasks into 3 arrays, EEG, ART, and EVT

%P300
p300_eeg_fnames=filenames(fullfile('*P300_Eyes_Closed.eeg'));
p300_art_fnames=filenames(fullfile('*P300_Eyes_Closed.art'));
p300_evt_fnames=filenames(fullfile('*P300_Eyes_closed.evt'));

%Flanker
flanker_eeg_fnames=filenames(fullfile('*Flanker_Test.eeg'));
flanker_art_fnames=filenames(fullfile('*Flanker_Test.art'));
flanker_evt_fnames=filenames(fullfile('*Flanker_Test.evt'));

%Chronic Back Pain Taskbackpain_art_fnames
backpain_eeg_fnames=filenames(fullfile('*EO_Baseline_12.eeg'));
backpain_art_fnames=filenames(fullfile('*EO_Baseline_12.art'));
backpain_evt_fnames=filenames(fullfile('*EO_Baseline_12.evt'));

%Resting State Back Pain
resting_eeg_fnames=filenames(fullfile('*EO_Baseline_8.eeg'));
resting_art_fnames=filenames(fullfile('*EO_Baseline_8.art'));
resting_evt_fnames=filenames(fullfile('*EO_Baseline_8.evt'));

%spit error if length all 3 fnames inconsistent

n=length(p300_eeg_fnames);

%%
%load in all the raw eeg data
for sub=1:n
    data{sub}=load(eeg_fnames{sub});
end

%%
%load in the evt files to make sure we're presenting 40 stims per person.
%The number of stimuli presented range from 79 to 82. There is also some
%variance in the total number of samples collected.
for sub=1:n
    evt{sub}=load(evt_fnames{sub});
end
for sub=1:n
    number_stimuli(sub,1)=size(find(evt{sub}(:)>0),1);
    number_stimuli(sub,2)=size(data{sub},1);
end

%%
%Removing artifacts
%load in art files first, find indices where artifacts occur (regardless of artifact amplitude)
%set values in data (raw eeg data) equal to 0 if an artifact occurs
% ^ maintains the original length of the collection.
for sub=1:n
    art{sub}=load(art_fnames{sub});
    art_index=find(art{sub}>0);
    %replace ART datapoints with zeros
    data{sub}(art_index)=0;
end

%%
%sanity check for consistent times between thumps
%vary, but almost always 0.6000 seconds (150 samples)
%0.5960 and 0.6040 appear often as well, so 1 sample error (?)
%also some 0s due to (?)
%sorry for the embedded loop :/
for sub=1:n
    numthumps=0;
    thump1=0;
    thump2=0;
    for sample=1:length(evt{sub})
        if evt{sub}(sample) == 1
            thump1=sample;
        end
        if evt{sub}(sample) == 2
            numthumps=numthumps+1;
            thump2=sample;
            thumplength=((thump2-thump1));
            thumps(numthumps,sub)=thumplength;
        end
    end
end

%%
%for every subject's SMS EEG file, split into 3 arrays
%1. setOne{sub}{thumpnum} = array of EEG data for 155 samples after each 1st thump
%each iteration should be a column = 155 samples
for sub=1:n
    onethump=find(evt{sub}==1)
    for thumpnum=1:length(onethump)
        try
            setOne{sub}{thumpnum}=data{sub}(onethump(thumpnum):onethump(thumpnum)+154,:)
        catch
            continue
        end
    end
end

%%
%2. setTwo{sub}{thumpnum} = array of EEG data for 155 samples after each 2nd thump
%each iteration should be a column = 155 samples
for sub=1:n
    twothump=find(evt{sub}==2)
    for thumpnum=1:length(twothump)
        try
            setTwo{sub}{thumpnum}=data{sub}(twothump(thumpnum):twothump(thumpnum)+154,:)
        catch
            continue
        end
    end
end
%%
%3. setThree{sub}{thumpnum} = array of EEG data until the next 1st thump
%each iteration should be a column of data, varying length
for sub=1:n
    for thumpnum=1:length(twothump)
        try
            setThree{sub}{thumpnum}=data{sub}(twothump(thumpnum)+150:onethump(thumpnum+1)+4,:)
        catch
            setThree{sub}{thumpnum}=data{sub}(twothump(thumpnum)+150:end,:)
        end
    end
end

%%
%4. manipulate setThree{sub}{thumpnum} such that it is curated into a new
%set of continuous 1s / 250 sample long matrices format
%restingState{sub}{contig}
%
% offset parameter should be used for creating a new set such that the
% start of the first window's start will be incremented by that many samples, and offset should be
% >0 and <250
offset = 0
windowLength = 250 %number of samples / window
for sub=1:n
    contig=1
    for thumpnum=1:length(setThree{sub})
        timeStamp=1
        while length(setThree{sub}{thumpnum}(timeStamp:end,:))>=250
            try
                restingState{sub}{contig}=setThree{sub}{thumpnum}(timeStamp:(timeStamp+windowLength-1),:)
                timeStamp=timeStamp+windowLength
                contig=contig+1
            catch
                contig=contig+1
                continue
            end
        end
    end
end



%%
% Listing out subject data types and counting
% 0 = control, 1 = pain, 2 = relief/follow-up
ncontrol=0;
npain=0;
nrelief=0;

orders=[
    {'Bannantine',2},
    {'Barker',1},
    {'Barker',2},
    {'Carter',0},
    {'Chambers',1},
    {'Chambers',2},
    {'Fosse',0},
    {'Gervasi',1},
    {'Gervasi',2},
    {'Gerwick',0},
    {'Gies',1},
    {'Gies',2},
    {'Kramer',1},
    {'Kramer',2},
    {'Mitchell',1},
    {'Neubauer',1},
    {'Neubauer',2},
    {'Oak',0},
    {'Olivas',1},
    {'Palosaari',1},
    {'Polosaari',2},
    {'Roth',2},
    {'Schneider',0},
    {'Simone',1},
    {'Sorbo',0}
    ];

for subject=1:length(orders)
    if orders{subject,2}==0
        ncontrol=ncontrol+1;
    elseif orders{subject,2}==1
        npain=npain+1;
    elseif orders{subject,2}==2
        nrelief=nrelief+1;
    end
end

%%
%Insert a column of order number into start of all resting state data
for sub=1:n
    for contig=1:length(restingState{sub})
        restingState{sub}{contig}=[repelem(orders{sub,2},length(restingState{sub}{contig}))' restingState{sub}{contig}]
    end
end

%%
%exporting data matrix as CSV file, numbered in alphabetical order
% YOU SHOULD CD INTO ANOTHER FOLDER BEFORE YOU RUN THIS SECTION
for sub=1:n
    if (sub~=2)&&(sub~=4)
        for contig=1:length(restingState{sub})
            if (orders{sub,2}==0) || (orders{sub,2}==1)
                %channel cols
                %csvwrite(strcat(num2str(sub),'_',num2str(contig),'.csv'),restingState{sub}{contig})
                %channel rows
                csvwrite(strcat(num2str(sub),'_inverse_',num2str(contig),'.csv'),restingState{sub}{contig}')
            end
        end
    end
end
