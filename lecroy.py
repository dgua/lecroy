#!/usr/bin/env python3
#
# This is a module for reading binary files from LeCroy scopes.
#
# LeCroy binary files have a .trc extension. This module reads version 'LECROY_2_3' of 
# the format (an exception is raised if a different format is encountered).
# 
# Dependencies: numpy, math.
#
# D. Guarisco, 2013-2018. Assembled from various sources.
#
# Version history:
#   1.0         2013-02-10  First release
#   1.1         2013-02-13  Added support for sequence acquisitions. Fixed a few bugs.
#   1.2         2013-07-30  Correctly handles the case where WAVEDESC block starts at the #                           beginning of the file (i.e., there is no file size info)
#   1.3         2013-08-27  Correctly imports FFT traces. TIMEBASE and FIXED VERT GAIN
#                           reflect horizontal and vertical units. 
#   1.4         2018-12-24  Ported to Python 3 (no longer works in Python 2). 
#                           Correctly reports FILE_SIZE in WAVEDESC.
#
import numpy as np
import math
#
#
def float2eng(f):
    """
    Converts a floating point number to its engineering representation using SI prefixes.
    """
    if f!=0: ex = math.floor(math.log(abs(f),10))
    else: ex = 0
    exeng = ex-ex % 3
    if exeng<-24: exeng = -24
    else: 
        if exeng>24: exeng = 24
    mant = f/(10**exeng)
    prefix = 'yzafpnum kMGTPEZY'
    id = int((exeng+24)/3)
    if id==8: up = ''
    else: up = prefix[id]    
    return "%g %s" % (mant,up)
#
#    
def ReadBinaryTrace(filePath):  
    """
    Reads a binary LeCroy file (file extension: .trc). Only version 'LECROY_2_3' of the 
    format is supported. A detailed description of the format can be obtained from the 
    instrument by issuing the command 'TEMPLATE?'
    
    The following results are returned:
        (WAVEDESC,USER_TEXT,x,y1,y2), where
        
    WAVEDESC is the file header, containing information about the waveform
    USER_TEXT is an optional string (empty by default)
    x  is a numpy array (dtype='float64') containing the time offsets of the waveform 
       relative to the trigger time
    y1 is a numpy array (dtype='float64') containing the primary waveform
    y2 is a numpy array (dtype='float64') containing the secondary waveform (empty for a
       single sweep acquisition).
       
    The shape of the arrays x and y depends on the mode of the scope during the data
    acquisition. If the scope is not in sequence mode (SUBARRAY_COUNT=1) a single sweep
    was captured. In this case, x and y1 are one-dimensional arrays of size
    WAVE_ARRAY_COUNT. If the scope was in sequence mode, several sweeps (SUBARRAY_COUNT) 
    were captured in sequence. In this case, the shape of x and y is a two-dimensional
    array(SUBARRAY_COUNT,WAVE_ARRAY_COUNT/SUBARRAY_COUNT).
    
    WAVEDESC is a dict with the keys shown below, which are exactly as described
    in the file template documentation, with the following exceptions:
    1. For enum types, an extra key (KEY_INDEX) has been added, containing the int
       index (starting at zero), as indicated in the list below
    2. TRIGGER_TIME is a tuple: (seconds:float64, minutes:int8, hours:int8, days:int8,
       months:int8, year:int16, unused:int16)
    3. FILE_SIZE (in bytes) has been added
    
        Parameter name                          Type
        DESCRIPTOR_NAME                         string(16)
        TEMPLATE_NAME                           string(16)
        COMM_TYPE_INDEX (enum index)            np.int16
        COMM_TYPE                               string
        COMM_ORDER_INDEX (enum index)           np.int16
        COMM_ORDER                              string
        WAVE_DESCRIPTOR                         np.int32
        USER_TEXT                               np.int32
        RES_DESC1                               np.int32
        TRIGTIME_ARRAY                          np.int32
        RIS_TIME_ARRAY                          np.int32
        RES_ARRAY1                              np.int32
        WAVE_ARRAY_1                            np.int32
        WAVE_ARRAY_2                            np.int32
        RES_ARRAY2                              np.int32
        RES_ARRAY3                              np.int32
        INSTRUMENT_NAME                         string(16)
        INSTRUMENT_NUMBER                       np.int32
        TRACE_LABEL                             string(16)
        RESERVED1                               np.int16
        RESERVED2                               np.int16
        WAVE_ARRAY_COUNT                        np.int32
        PNTS_PER_SCREEN                         np.int32
        FIRST_VALID_PNT                         np.int32
        LAST_VALID_PNT                          np.int32
        FIRST_POINT                             np.int32
        SPARSING_FACTOR                         np.int32
        SEGMENT_INDEX                           np.int32
        SUBARRAY_COUNT                          np.int32
        SWEEPS_PER_ACQ                          np.int32
        POINTS_PER_PAIR                         np.int16
        PAIR_OFFSET                             np.int16
        VERTICAL_GAIN                           np.float32
        VERTICAL_OFFSET                         np.float32
        MAX_VALUE                               np.float32
        MIN_VALUE                               np.float32
        NOMINAL_BITS                            np.int16
        NOM_SUBARRAY_COUNT                      np.int16
        HORIZ_INTERVAL                          np.float32
        HORIZ_OFFSET                            np.float64
        PIXEL_OFFSET                            np.float64
        VERTUNIT                                string(48)
        HORUNIT                                 string(48)
        HORIZ_UNCERTAINTY                       np.float32
        TRIGGER_TIME                            tuple
        ACQ_DURATION                            np.float32
        RECORD_TYPE_INDEX (enum index)          np.int16
        RECORD_TYPE                             string
        PROCESSING_DONE_INDEX (enum index)      np.int16
        PROCESSING_DONE                         string
        RESERVED5                               np.int16
        RIS_SWEEPS                              np.int16
        TIMEBASE_INDEX (enum index)             np.int.16
        TIMEBASE                                string
        VERT_COUPLING_INDEX                     np.int16
        VERT_COUPLING                           string
        PROBE_ATT                               np.float32
        FIXED_VERT_GAIN_INDEX (enum index)      np.int16
        FIXED_VERT_GAIN                         string
        BANDWIDTH_LIMIT_INDEX (enum index)      np.int16
        BANDWIDTH_LIMIT                         string
        VERTICAL_VERNIER                        np.float32
        ACQ_VERT_OFFSET                         np.float32
        WAVE_SOURCE_INDEX (enum index)          np.int16
        WAVE_SOURCE                             string
        FILE_SIZE                               int
    """
    # Open binary data file for read
    with open(filePath, 'rb') as dataFile:           
        # Read wave descriptor block
        str2 = dataFile.read(32)           
        startOffset = str2.find(b'WAVEDESC')
        if startOffset==-1: 
            raise RuntimeError('File is not in a recognizable format')
            return
        DESCRIPTOR_NAME = str2[startOffset:startOffset+16].rstrip(b'\0x00').decode('utf-8')
        # Try to read file size, if present
        try:
            fileSize = int.from_bytes(str2[4:startOffset],byteorder='little') + startOffset
        except ValueError:
            fileSize = 0
        #Find COMM_ORDER first. Use this byte order in all subsequent reads from file
        dataFile.seek(startOffset+34)
        COMM_ORDER_INDEX = np.fromfile(dataFile,dtype='<i2',count=1)[0]
        COMM_ORDER = ('HIFIRST','LOFIRST')[COMM_ORDER_INDEX]
        if COMM_ORDER == 'LOFIRST': co = '<'
        else: co = '>'
        #Now read all other header parameters           
        dataFile.seek(startOffset+16)
        TEMPLATE_NAME = dataFile.read(16).rstrip(b'\0x00').decode('utf-8')
        if TEMPLATE_NAME != 'LECROY_2_3':
            raise RuntimeError('Template version different from LECROY_2_3')
            return
        COMM_TYPE_INDEX = np.fromfile(dataFile,dtype=co+'i2',count=1)[0]
        COMM_TYPE = ('byte','word')[COMM_TYPE_INDEX]
        dataFile.seek(startOffset+36)
        WAVE_DESCRIPTOR = np.fromfile(dataFile,dtype=co+'i4',count=1)[0]
        USER_TEXT = np.fromfile(dataFile,dtype=co+'i4',count=1)[0]
        RES_DESC1 = np.fromfile(dataFile,dtype=co+'i4',count=1)[0]           
        TRIGTIME_ARRAY = np.fromfile(dataFile,dtype=co+'i4',count=1)[0]   
        RIS_TIME_ARRAY = np.fromfile(dataFile,dtype=co+'i4',count=1)[0]
        RES_ARRAY1 = np.fromfile(dataFile,dtype=co+'i4',count=1)[0]
        WAVE_ARRAY_1 = np.fromfile(dataFile,dtype=co+'i4',count=1)[0]          
        WAVE_ARRAY_2 = np.fromfile(dataFile,dtype=co+'i4',count=1)[0]
        RES_ARRAY2 = np.fromfile(dataFile,dtype=co+'i4',count=1)[0]
        RES_ARRAY3 = np.fromfile(dataFile,dtype=co+'i4',count=1)[0]
        INSTRUMENT_NAME = dataFile.read(16).rstrip(b'\x00').decode('utf-8')
        INSTRUMENT_NUMBER = np.fromfile(dataFile,dtype=co+'i4',count=1)[0]
        TRACE_LABEL = dataFile.read(16).rstrip(b'\x00').decode('utf-8')
        RESERVED1 = np.fromfile(dataFile,dtype=co+'i2',count=1)[0]
        RESERVED2 = np.fromfile(dataFile,dtype=co+'i2',count=1)[0]
        WAVE_ARRAY_COUNT = np.fromfile(dataFile,dtype=co+'i4',count=1)[0]
        PNTS_PER_SCREEN = np.fromfile(dataFile,dtype=co+'i4',count=1)[0]
        FIRST_VALID_PNT = np.fromfile(dataFile,dtype=co+'i4',count=1)[0]
        LAST_VALID_PNT = np.fromfile(dataFile,dtype=co+'i4',count=1)[0]
        FIRST_POINT = np.fromfile(dataFile,dtype=co+'i4',count=1)[0]
        SPARSING_FACTOR = np.fromfile(dataFile,dtype=co+'i4',count=1)[0]
        SEGMENT_INDEX = np.fromfile(dataFile,dtype=co+'i4',count=1)[0]
        SUBARRAY_COUNT = np.fromfile(dataFile,dtype=co+'i4',count=1)[0]
        SWEEPS_PER_ACQ = np.fromfile(dataFile,dtype=co+'i4',count=1)[0]
        POINTS_PER_PAIR = np.fromfile(dataFile,dtype=co+'i2',count=1)[0]
        PAIR_OFFSET = np.fromfile(dataFile,dtype=co+'i2',count=1)[0]
        VERTICAL_GAIN = np.fromfile(dataFile,dtype=co+'f4',count=1)[0]
        VERTICAL_OFFSET = np.fromfile(dataFile,dtype=co+'f4',count=1)[0]
        MAX_VALUE = np.fromfile(dataFile,dtype=co+'f4',count=1)[0]
        MIN_VALUE = np.fromfile(dataFile,dtype=co+'f4',count=1)[0]
        NOMINAL_BITS = np.fromfile(dataFile,dtype=co+'i2',count=1)[0]
        NOM_SUBARRAY_COUNT = np.fromfile(dataFile,dtype=co+'i2',count=1)[0]
        HORIZ_INTERVAL = np.fromfile(dataFile,dtype=co+'f4',count=1)[0]
        HORIZ_OFFSET = np.fromfile(dataFile,dtype=co+'f8',count=1)[0]
        PIXEL_OFFSET = np.fromfile(dataFile,dtype=co+'f8',count=1)[0]
        VERTUNIT = dataFile.read(48).rstrip(b'\0x00').decode('utf-8')
        HORUNIT = dataFile.read(48).rstrip(b'\0x00').decode('utf-8')
        HORIZ_UNCERTAINTY = np.fromfile(dataFile,dtype=co+'f4',count=1)[0]
        TRIGGER_TIME_SECONDS = np.fromfile(dataFile,dtype=co+'f8',count=1)[0]
        TRIGGER_TIME_MINUTES = np.fromfile(dataFile,dtype=np.int8,count=1)[0]
        TRIGGER_TIME_HOURS = np.fromfile(dataFile,dtype=np.int8,count=1)[0]
        TRIGGER_TIME_DAYS = np.fromfile(dataFile,dtype=np.int8,count=1)[0]
        TRIGGER_TIME_MONTHS = np.fromfile(dataFile,dtype=np.int8,count=1)[0]
        TRIGGER_TIME_YEAR = np.fromfile(dataFile,dtype=co+'i2',count=1)[0]
        TRIGGER_TIME_UNUSED = np.fromfile(dataFile,dtype=co+'i2',count=1)[0]
        ACQ_DURATION = np.fromfile(dataFile,dtype=co+'f4',count=1)[0]
        RECORD_TYPE_INDEX = np.fromfile(dataFile,dtype=co+'i2',count=1)[0]
        RECORD_TYPE = ('single sweep','interleaved','histogram','graph','filter_coefficient','complex','extrema','sequence obsolete','centered RIS','peak detect')[RECORD_TYPE_INDEX]
        PROCESSING_DONE_INDEX = np.fromfile(dataFile,dtype=co+'i2',count=1)[0]
        PROCESSING_DONE = ('no processing','fir filter','interpolated','sparsed','autoscaled','no result','rolling','cumulative')[PROCESSING_DONE_INDEX]
        RESERVED5 = np.fromfile(dataFile,dtype=co+'i2',count=1)[0]
        RIS_SWEEPS = np.fromfile(dataFile,dtype=co+'i2',count=1)[0]
        TIMEBASE_INDEX = np.fromfile(dataFile,dtype=co+'i2',count=1)[0]
        if TIMEBASE_INDEX==100: TIMEBASE = 'EXTERNAL'
        else:
            divisions = (1,2,5)
            mant = divisions[TIMEBASE_INDEX % 3]
            exp = int(TIMEBASE_INDEX / 3)-12
            t = mant*10**exp
            TIMEBASE = float2eng(t)+HORUNIT+'/div'           
        VERT_COUPLING_INDEX = np.fromfile(dataFile,dtype=co+'i2',count=1)[0]
        VERT_COUPLING = ('DC 50 Ohms','ground','DC 1MOhm','ground','AC 1MOhm')[VERT_COUPLING_INDEX]
        PROBE_ATT = np.fromfile(dataFile,dtype=co+'f4',count=1)[0]
        FIXED_VERT_GAIN_INDEX = np.fromfile(dataFile,dtype=co+'i2',count=1)[0]
        mant = divisions[FIXED_VERT_GAIN_INDEX % 3]
        exp = int(FIXED_VERT_GAIN_INDEX / 3)-6
        t = mant*10**exp
        FIXED_VERT_GAIN = float2eng(t)+VERTUNIT+'/div'
        BANDWIDTH_LIMIT_INDEX = np.fromfile(dataFile,dtype=co+'i2',count=1)[0]
        BANDWIDTH_LIMIT = ('off','on')[BANDWIDTH_LIMIT_INDEX]
        VERTICAL_VERNIER = np.fromfile(dataFile,dtype=co+'f4',count=1)[0]
        ACQ_VERT_OFFSET = np.fromfile(dataFile,dtype=co+'f4',count=1)[0]
        WAVE_SOURCE_INDEX = np.fromfile(dataFile,dtype=co+'i2',count=1)[0]  
        WAVE_SOURCE = 'C%d' % (WAVE_SOURCE_INDEX + 1)
        #Sanity check
        coffset = dataFile.tell()-startOffset
        if coffset != WAVE_DESCRIPTOR:
            raise RuntimeError('Wave descriptor is too long!')
            return           
        #Create WAVEDESC dict
        WAVEDESC = {'DESCRIPTOR_NAME':DESCRIPTOR_NAME}        
        WAVEDESC['TEMPLATE_NAME'] = TEMPLATE_NAME
        WAVEDESC['COMM_TYPE_INDEX'] = COMM_TYPE_INDEX
        WAVEDESC['COMM_TYPE'] = COMM_TYPE
        WAVEDESC['COMM_ORDER_INDEX'] = COMM_ORDER_INDEX
        WAVEDESC['COMM_ORDER'] = COMM_ORDER
        WAVEDESC['WAVE_DESCRIPTOR'] = WAVE_DESCRIPTOR
        WAVEDESC['USER_TEXT'] = USER_TEXT
        WAVEDESC['RES_DESC1'] = RES_DESC1
        WAVEDESC['TRIGTIME_ARRAY'] = TRIGTIME_ARRAY
        WAVEDESC['RIS_TIME_ARRAY'] = RIS_TIME_ARRAY
        WAVEDESC['RES_ARRAY1'] = RES_ARRAY1
        WAVEDESC['WAVE_ARRAY_1'] = WAVE_ARRAY_1
        WAVEDESC['WAVE_ARRAY_2'] = WAVE_ARRAY_2
        WAVEDESC['RES_ARRAY2'] = RES_ARRAY2
        WAVEDESC['RES_ARRAY3'] = RES_ARRAY3
        WAVEDESC['INSTRUMENT_NAME'] = INSTRUMENT_NAME
        WAVEDESC['INSTRUMENT_NUMBER'] = INSTRUMENT_NUMBER
        WAVEDESC['TRACE_LABEL'] = TRACE_LABEL
        WAVEDESC['RESERVED1'] = RESERVED1
        WAVEDESC['RESERVED2'] = RESERVED2
        WAVEDESC['WAVE_ARRAY_COUNT'] = WAVE_ARRAY_COUNT
        WAVEDESC['PNTS_PER_SCREEN'] = PNTS_PER_SCREEN
        WAVEDESC['FIRST_VALID_PNT'] = FIRST_VALID_PNT
        WAVEDESC['LAST_VALID_PNT'] = LAST_VALID_PNT
        WAVEDESC['FIRST_POINT'] = FIRST_POINT
        WAVEDESC['SPARSING_FACTOR'] = SPARSING_FACTOR
        WAVEDESC['SEGMENT_INDEX'] = SEGMENT_INDEX
        WAVEDESC['SUBARRAY_COUNT'] = SUBARRAY_COUNT
        WAVEDESC['SWEEPS_PER_ACQ'] = SWEEPS_PER_ACQ
        WAVEDESC['POINTS_PER_PAIR'] = POINTS_PER_PAIR
        WAVEDESC['PAIR_OFFSET'] = PAIR_OFFSET
        WAVEDESC['VERTICAL_GAIN'] = VERTICAL_GAIN
        WAVEDESC['VERTICAL_OFFSET'] = VERTICAL_OFFSET
        WAVEDESC['MAX_VALUE'] = MAX_VALUE
        WAVEDESC['MIN_VALUE'] = MIN_VALUE
        WAVEDESC['NOMINAL_BITS'] = NOMINAL_BITS
        WAVEDESC['NOM_SUBARRAY_COUNT'] = NOM_SUBARRAY_COUNT
        WAVEDESC['HORIZ_INTERVAL'] = HORIZ_INTERVAL
        WAVEDESC['HORIZ_OFFSET'] = HORIZ_OFFSET
        WAVEDESC['PIXEL_OFFSET'] = PIXEL_OFFSET
        WAVEDESC['VERTUNIT'] = VERTUNIT
        WAVEDESC['HORUNIT'] = HORUNIT
        WAVEDESC['HORIZ_UNCERTAINTY'] = HORIZ_UNCERTAINTY
        WAVEDESC['TRIGGER_TIME'] = (TRIGGER_TIME_SECONDS,TRIGGER_TIME_MINUTES,TRIGGER_TIME_HOURS,TRIGGER_TIME_DAYS,TRIGGER_TIME_MONTHS,TRIGGER_TIME_YEAR,TRIGGER_TIME_UNUSED)
        WAVEDESC['ACQ_DURATION'] = ACQ_DURATION
        WAVEDESC['RECORD_TYPE_INDEX'] = RECORD_TYPE_INDEX
        WAVEDESC['RECORD_TYPE'] = RECORD_TYPE
        WAVEDESC['PROCESSING_DONE_INDEX'] = PROCESSING_DONE_INDEX
        WAVEDESC['PROCESSING_DONE'] = PROCESSING_DONE
        WAVEDESC['RESERVED5'] = RESERVED5
        WAVEDESC['RIS_SWEEPS'] = RIS_SWEEPS
        WAVEDESC['TIMEBASE_INDEX'] = TIMEBASE_INDEX
        WAVEDESC['TIMEBASE'] = TIMEBASE
        WAVEDESC['VERT_COUPLING_INDEX'] = VERT_COUPLING_INDEX
        WAVEDESC['VERT_COUPLING'] = VERT_COUPLING
        WAVEDESC['PROBE_ATT'] = PROBE_ATT
        WAVEDESC['FIXED_VERT_GAIN_INDEX'] = FIXED_VERT_GAIN_INDEX
        WAVEDESC['FIXED_VERT_GAIN'] = FIXED_VERT_GAIN    
        WAVEDESC['BANDWIDTH_LIMIT_INDEX'] = BANDWIDTH_LIMIT_INDEX
        WAVEDESC['BANDWIDTH_LIMIT'] = BANDWIDTH_LIMIT
        WAVEDESC['VERTICAL_VERNIER'] = VERTICAL_VERNIER
        WAVEDESC['ACQ_VERT_OFFSET'] = ACQ_VERT_OFFSET
        WAVEDESC['WAVE_SOURCE_INDEX'] = WAVE_SOURCE_INDEX
        WAVEDESC['WAVE_SOURCE'] = WAVE_SOURCE
        WAVEDESC['FILE_SIZE'] = fileSize
 
         #Read user text (160 char. maximum)
        if USER_TEXT >0: TEXT = dataFile.read(USER_TEXT)
        else: TEXT = b''
            
        #Read waveforms. Distinguish case of acquisition sequence or single acquisition
        if SUBARRAY_COUNT>1:
            #Multiple segments
            #Sanity check
            if WAVE_ARRAY_COUNT % SUBARRAY_COUNT !=0:
                raise RuntimeError('Number of data points is not a multiple of number of segments')
                return  
            npts = WAVE_ARRAY_COUNT // SUBARRAY_COUNT
            #Read TRIGTIME array first. There are SUBARRAY_COUNT repetitions of two doubles, TRIGGER_TIME and TRIGGER_OFFSET
            record_type = np.dtype([('TRIGGER_TIME', co+'f8'),('TRIGGER_OFFSET',co+'f8')])
            trigArray = np.fromfile(dataFile,dtype=record_type,count=SUBARRAY_COUNT)
            trigTime = trigArray['TRIGGER_TIME']
            trigOffset = trigArray['TRIGGER_OFFSET']
            #Generate trigger time array
            x = np.zeros((SUBARRAY_COUNT,npts),dtype='float64')

            for i in range(SUBARRAY_COUNT):
                horOffset = trigTime[i]+trigOffset[i]
                x[i:] = np.arange(npts,dtype='float64')*HORIZ_INTERVAL+horOffset
            #Now read data array
            if COMM_TYPE_INDEX==0:
                y1 = np.fromfile(dataFile,dtype=co+'i1',count=WAVE_ARRAY_COUNT).reshape(SUBARRAY_COUNT,npts)*VERTICAL_GAIN-VERTICAL_OFFSET
            else:
                y1 = np.fromfile(dataFile,dtype=co+'i2',count=WAVE_ARRAY_COUNT).reshape(SUBARRAY_COUNT,npts)*VERTICAL_GAIN-VERTICAL_OFFSET
            y2 = np.array([])
        else:
            #Single sweep. Read waveforms from file
            if COMM_TYPE_INDEX==0:
                y1 = np.fromfile(dataFile,dtype=co+'i1',count=WAVE_ARRAY_COUNT)*VERTICAL_GAIN-VERTICAL_OFFSET
                if WAVE_ARRAY_2>0:
                    y2 = np.fromfile(dataFile,dtype=co+'i1',count=WAVE_ARRAY_COUNT)*VERTICAL_GAIN-VERTICAL_OFFSET
                else:
                    y2 = np.array([])
            else:
                y1 = np.fromfile(dataFile,dtype=co+'i2',count=WAVE_ARRAY_COUNT)*VERTICAL_GAIN-VERTICAL_OFFSET
                if WAVE_ARRAY_2>0:
                    y2 = np.fromfile(dataFile,dtype=co+'i2',count=WAVE_ARRAY_COUNT)*VERTICAL_GAIN-VERTICAL_OFFSET
                else:
                    y2 = np.array([])
            #Generate time intervals
            x = np.arange(WAVE_ARRAY_COUNT,dtype='float64')*HORIZ_INTERVAL+HORIZ_OFFSET   
        return WAVEDESC,TEXT,x,y1,y2


if __name__ == '__main__':
    print('lecroy module to read binary LeCroy files. Type help(lecroy.ReadBinaryTrace) for more info')