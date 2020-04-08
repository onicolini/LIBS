%OOcombine
%Combine scan-files from Ocean Optics spectrometer to matrix.
%This is saved either as mat-file, text-file, or Excel-file
%
%If put in a mat-file only about 12% storage is needed compared with the
%original OO-files.
%
%Jonas Petersson, Swerea

%% Settings
plot_spec='no'; %('plot'/'no') plot the spectra
rev_opt='no'; %('rev'/'no') asked if to reverse order of scans
excel_write='no'; %('xls'/'no') %write file also in Excel format

%% Find the files
[files,fold]=uigetfile('*.*','Select file','MultiSelect','on');

if not(iscell(files))
    files={files};
end
    
Nf=length(files);

%% Read the files and combine
for f=1:Nf
    file=files{f};
    finfo=dir(fullfile(fold,file));%(to get the file-size)
    % number of rows are estimated given the known number of columns
%     Nscans=round((finfo.bytes/18654-2));
    Nscans=round((finfo.bytes/18400-2));
    disp(['Reading file ' num2str(f) '/' num2str(Nf), ', #scans ~' ...
        num2str(Nscans)])
    %Read the first file and put in vector 'data'
    fid=fopen(fullfile(fold,file));
    fgetl(fid); %first row is the pixels
    a=fgetl(fid);
    a(strfind(a,','))='.';
    wl=str2num(a(11:end))';
    wl=wl(1:end-1); % it can be -2 depending on the number of pixels that are used in the spectra, see when importing raw datafiles in matlab, how many pixels are used
    
    temp=zeros(Nscans,length(wl));
    savetime=zeros(1,Nscans);
    timestamp=zeros(1,Nscans);
    
    i=1;
    a=fgetl(fid);
    fprintf('Reading scan:    0')
    while not(isequal(a,-1))
        a(strfind(a,','))='.';
        i=i+1;
       % temp(i,:)=str2num(a(30:end));
      temp(i,:)=str2num(a(38:end));
        savetime(i)=datenum(a(1:19));
      %timestamp(i)=str2double(a(21:29));
      timestamp(i)=str2double(a(21:37));
        fprintf('\b\b\b\b')
        fprintf('%4.0f',i)
        a=fgetl(fid);
    end
    fclose(fid);
    fprintf('\n')
    
    %put the first spectra in the matrix 'A'
    N=i;
    data=temp(1:N,:)';
    savetime=savetime(1:N);
    timestamp=timestamp(1:N);
    
    [~, filename, ~]=fileparts(file);
    
    %% Plot the spectra
    N=length(data(1,:));
    if strcmp(plot_spec,'plot')
        plotfig=figure('name',filename);
        coloring=colormap(jet(N));
        % plot(data(:,1),zeros(size(data(:,1))),'color',[.5 .5 .5],...
        %     'displayname','baseline')
        hold on
        for i=1:N
            plot(wl,data(:,i),'color',coloring(i,:),...
                'displayname',num2str(i))
        end
        legend('show')
        set(legend,'interpreter','none')
    end
    
    save([fullfile(fold,filename) '.mat'],'data','wl','savetime','timestamp')
    disp(['file saved as ' fullfile(fold,filename) '.mat'])
    
    
    % %% Reverse order
    % if strcmp(rev_opt,'rev')
    %     rev=input('Reverse order? (y/n) ','s');
    %     if strcmp(rev,'y') || strcmp(rev,'1')
    %         %reverse order
    %         data_rev=zeros(size(A));
    %         A_rev(:,1)=A(:,1);
    %         for i=1:N
    %             A_rev(:,i+1)=A(:,N+2-i);
    %         end
    %
    %         figure(plotfig);
    %         clf(plotfig);
    %         coloring=colormap(jet(N));
    %         hold on
    %         for i=1:N
    %             plot(A_rev(:,1),A_rev(:,i+1),'color',coloring(i,:),...
    %                 'displayname',char(files(N+1-i)))
    %         end
    %         legend('show')
    %         set(legend,'interpreter','none')
    %         A=A_rev;
    %     end
    % end
    
    %% Save file
    % [filename, pathname, type] = uiputfile({'*.mat','MAT-file (*.mat)'; ...
    %     '*.txt','Text-file (*.txt)'; '*.xls',  'Excel-file (*.xls)'},...
    %     'Save spectra to file',[filefold '.mat']);
    %
    % if not(isequal(filename,0)) && not(isequal(pathname,0))
    %     if type==1
    %         wl=A(:,1);
    %         data=A(:,2:end);
    %         save(fullfile(pathname,filename),'filetext','data','wl')
    %         disp('Fieldnames: ''filetext'',''data'',''wl''')
    %     elseif type==2
    %         save(fullfile(pathname,filename),'A','-ascii')
    %     elseif type==3
    %         xlswrite(fullfile(pathname,filename),A);
    %     end
    %     if type==1 || type==2 || type==3
    %         disp(['file saved as ' fullfile(pathname,filename)])
    %     else
    %         disp('File not saved')
    %     end
    %
    %     if strcmp(excel_write,'xls') && type~=3
    %         %Save a copy in Excel. For Arne... :)
    %         del=findstr(filename,'.');
    %         ext=filename(del(end)+1:end);
    %         name_xls=filename(1:del(end)-1);
    %         xlswrite(fullfile(pathname,name_xls),A);
    %         disp(['also saved in .xls file: ' name_xls '.xls'])
    %     end
    %
    % else
    %     disp('file not saved')
    % end
end
