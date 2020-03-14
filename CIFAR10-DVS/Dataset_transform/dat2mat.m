function [CIN]=dat2mat(file);
%
% loads events from a jAER .dat/.aedat file.
% allAddr are int16 raw addresses.
% allTs are int32 timestamps (1 us tick).
% noarg invocations open file browser dialog (in the case of no input argument)
% directly create vars allAddr, allTs in
% base workspace (in the case of no output argument).
%
% Header lines starting with '#' are ignored and printed
%
% Note: it is possible that the header parser can be fooled if the first
% data byte is the comment character '#'; in this case the header must be
% manually removed before parsing. Each header line starts with '#' and
% ends with the hex characters 0x0D 0x0A (CRLF, windows line ending).
%
% This functions creates a 6 column matlab matrix, where each line
% represents one event. It expects a jAER recorded file which displays
% under filters RetinaTeresa2, tmpdiff128, DVS128.
% Basically, this means that the 16 bits recorded from a physical AER parallel
% bus represent the following:
% bit 0: polarity
% bits 1 to 7:  x coordinate (from 0 to 127)
% bits 8 to 14: y coordinate (from 0 to 127)
% bit 15: ignored.
% This distribution of bits can be changed below.
%
% Each line in the matrix represents a recorded event.
% The 6 columns of the input matrix mean the following:
%
% Column 1: timestamps with 1us time tick
% Columns 2-3: ignore them (they are meant for simulator AERST (see Perez-Carrasco et al, IEEE
% TPAMI, Nov. 2013)).
% Column 4: x coordinate (from 0 to 127)
% Column 5: y coordinate (from 0 to 127)
% Column 6: event polarity
% 
% Originally written by Tobi Delbruck during CAVIAR project.
% Adapted by José A. Pérez-Carrasco in 2007, while at the Sevilla
% Microelectronics Institute (IMSE-CNM, CSIC and Univ. of Sevilla, Spain).
%

maxEvents=30e6;

if nargin==0
    [filename,path,filterindex] = uigetfile('*.aedat;*.dat','Select recorded retina data file');
    if filename==0, return; end
end
if nargin==1
    path='';
    filename=file;
end


f=fopen([path,filename],'r');
% skip header lines
bof=ftell(f);
line=native2unicode(fgets(f));
tok='#!AER-DAT';
version=0;

while line(1)=='#'
    if strncmp(line,tok, length(tok))==1
        version=sscanf(line(length(tok)+1:end),'%f');
    end
%     fprintf('%s\n',line(1:end-2)); % print line using \n for newline, discarding CRLF written by java under windows
    bof=ftell(f);
    line=native2unicode(fgets(f)); % gets the line including line ending chars
end
version2 = version;
% switch version,
%     case 0
% %         fprintf('No #!AER-DAT version header found, assuming 16 bit addresses\n');
%         version=1;
%     case 1
% %         fprintf('Addresses are 16 bit\n');
%     case 2
% %         fprintf('Addresses are 32 bit\n');
%     otherwise
% %         fprintf('Unknown file version %g',version);
% end

numBytesPerEvent=6;
switch(version)
    case 1
        numBytesPerEvent=6;
    case 2
        numBytesPerEvent=8;
end

        
fseek(f,0,'eof');
numEvents=floor((ftell(f)-bof)/numBytesPerEvent); % 6 bytes/event
if numEvents>maxEvents
%     fprintf('clipping to %d events although there are %d events in file\n',maxEvents,numEvents);
    numEvents=maxEvents;
end

% read data
fseek(f,bof,'bof'); % start just after header
switch version
    case 1
        allAddr=uint16(fread(f,numEvents,'uint16',4,'b')); % addr are each 2 bytes (uint16) separated by 4 byte timestamps
        fseek(f,bof+2,'bof'); % timestamps start 2 after bof
        allTs=uint32(fread(f,numEvents,'uint32',2,'b')); % ts are 4 bytes (uint32) skipping 2 bytes after each
    case 2
        allAddr=uint32(fread(f,numEvents,'uint32',4,'b')); % addr are each 4 bytes (uint32) separated by 4 byte timestamps
        fseek(f,bof+4,'bof'); % timestamps start 4 after bof
        allTs=uint32(fread(f,numEvents,'uint32',4,'b')); % ts are 4 bytes (uint32) skipping 4 bytes after each
end

fclose(f);

if nargout==0
   assignin('base','allAddr',allAddr);
   assignin('base','allTs',allTs);
%    fprintf('%d events assigned in base workspace as allAddr,allTs\n', length(allAddr));
%    dt=allTs(end)-allTs(1);
%    fprintf('min addr=%d, max addr=%d, Ts0=%d, deltaT=%d=%.2f s assuming 1 us timestamps\n',...
%        min(allAddr), max(allAddr), allTs(1), dt,double(dt)/1e6);
end

tpo=allTs;

kk=find(tpo(1:end-1)>tpo(2:end));
if ~isempty(kk)

    for i=1:length(kk)-1
        tpo(kk(i)+1:kk(i+1))=tpo(kk(i))*ones(size(tpo(kk(i)+1:kk(i+1)))) + tpo(kk(i)+1:kk(i+1));
    end
    tpo(kk(end)+1:end)=tpo(kk(end))*ones(size(tpo(kk(end)+1:end))) + tpo(kk(end)+1:end);

end

e=find(allAddr<=0);
allAddr(e)=0;

retinaSizeX=128;

% The following lines map x,y, and event polarity into specific bits of a
% 2-byte word for displaying with jAER's filters tmpdiff128 or
% RetinaTeresa2. You can change these bits according to the jAER filters
% you may want to use.
persistent xmask ymask xshift yshift polmask
if isempty(xmask)
    xmask = hex2dec ('fE'); % x are 7 bits (64 cols) ranging from bit 1-7
    ymask = hex2dec ('7f00'); % y are also 7 bits ranging from bit 8 to 14.
    xshift=1; % bits to shift x to right
    yshift=8; % bits to shift y to right
    polmask=1; % polarity bit is LSB
end


%mask aer addresses to ON and OFF address-strings
% find spikes in frame window
% if any(addr<0), warning('negative address'); end

addr=abs(allAddr); % make sure nonnegative or an error will result from bitand (glitches can somehow result in negative addressses...)
x=retinaSizeX-1-double(bitshift(bitand(addr,xmask),-xshift)); % x addresses
y=double(bitshift(bitand(addr,ymask),-yshift)); % y addresses
pol=1-2*double(bitand(addr,polmask)); % 1 for ON, -1 for OFF

tpo(:)=tpo(:)-tpo(1);

CIN=[tpo x y pol];


