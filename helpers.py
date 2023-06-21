from datetime import timedelta

def remove_chattering_alarms(alarm_dict_list, column_name = "SourceName_Identifier", count_threshold = 3):
    """
    This function removes the chattering alarms that occurs more than 3 times in 1 min
    Returns: alarm_dict_list without chattering alarms
    """
    filtered_alarm_list = []
    count_dict = {}
    prev_alarm = None
    for alarm in alarm_dict_list:
        if prev_alarm is not None and (alarm["StartTime"] - prev_alarm["EndTime"]) <= timedelta(minutes=1):
            count_dict[alarm[column_name]] = count_dict.get(alarm[column_name], 0) + 1
            if count_dict[alarm[column_name]] <= count_threshold:
                filtered_alarm_list.append(alarm)
        else:
            filtered_alarm_list.append(alarm)
            count_dict = {}
        prev_alarm = alarm
    return filtered_alarm_list


# create sequence_segmentation function
def sequence_segmentation(alarm_dict_list, time_delta):
    """
    This function segments the alarm sequence according to the time_delta.
    alarm_dict_list: list of dictionaries
    time_delta: time delta in seconds
    Returns: list of lists (alarms seperated by time_delta)
    """
    alarm_sequence = []
    for i in range(len(alarm_dict_list)):
        if i == 0:
            alarm_sequence.append([alarm_dict_list[i]])
        else:
            if alarm_dict_list[i]["TimeDelta"] < time_delta: # add this alarm into a new segment
                alarm_sequence[-1].append(alarm_dict_list[i])
            else:
                alarm_sequence.append([alarm_dict_list[i]])
    return alarm_sequence

def sequence_lst(alarm_sequence, alarm_definition = "SourceName_Identifier", min_seq_len = 5):
    """
    This function converts alarm_sequence into a list of strings
    alarm_sequence: list of lists (alarms seperated by time_delta)
    alarm_definition: alarm definition
    min_seq_len: minimum sequence length
    Returns: list of strings
    """
    seq_lst = []
    for sequence in alarm_sequence:
        seq = " ".join([alarm[alarm_definition] for alarm in sequence])
        
        if len(seq.split(" ")) > min_seq_len:
            seq_lst.append(seq)
    return seq_lst

