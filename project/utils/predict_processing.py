# coding=utf-8
from model.header import Header

def get_header_list(khoa, thuat_toan):
    print("iden: ", khoa + "_" + thuat_toan)
    data = {
        "identify": khoa + "_" + thuat_toan
    }
    header_list = Header.find_one(query=data)
    if header_list == -1:
        return []
    return header_list


def Clean_unecessary_subject(predict_subjects, trainning_subjects, result_list):
    unecessary_subject = []
    cleaned_subject_list = []
    clean_result_list = result_list
    print("predict_subjects: ", len(clean_result_list))
    for index, subject in enumerate(predict_subjects):
        if subject not in trainning_subjects:
            print("subject: ", subject)
            unecessary_subject.append(subject)
            clean_result_list.pop(index-len(unecessary_subject))
        else:
            cleaned_subject_list.append(subject)
    print("okke 1")
    print("cleaned_subject_list: ", len(cleaned_subject_list))
    print("unecessary_subject: ", unecessary_subject)
    return unecessary_subject, cleaned_subject_list, clean_result_list

def Add_incompliance_subject_to_predict(predict_subjects, trainning_subjects, result_list):
    incompliance_subject = []
    full_subject_list = predict_subjects
    full_result_list = result_list
    print("subject: ", full_subject_list)
    for subject in trainning_subjects:
        if subject not in predict_subjects:
            print("thieu")
            full_subject_list.append(subject)
            full_result_list.append(0)
            incompliance_subject.append(subject)
        else:
            continue
    print("okke 2")
    return incompliance_subject, full_subject_list, full_result_list


def Sort_predict_subject_list(predict_subjects, trainning_subjects, result_list):
    correct_index = []
    print("trainning_subjects: ", trainning_subjects)
    print("result_list: ", result_list)
    # tìm thứ tự đúng
    for element, subject in enumerate(trainning_subjects):
        if subject == predict_subjects[element]:
            correct_index.append(element)
        else:
            for header_element, header_subject in enumerate(predict_subjects):
                if header_subject == subject:
                    correct_index.append(header_element)
                    break
                else:
                    continue
    print("result_list: ", result_list)
    print(len(result_list))
    print(len(correct_index))
    stop_change = []
    for number, mark in enumerate(result_list):
        if number == correct_index[number]:
            continue
        else:
            if number in stop_change:
                continue
            else:

                temp = mark
                temp1 = result_list[correct_index[number]]

                result_list[number] = temp1,
                
                result_list[number] = int(result_list[number][0])
                result_list[correct_index[number]] = temp

                stop_change.append(correct_index[number])
    print("okke 3: ", len(result_list))                    
    return result_list

def VerifyAndChangeData(khoa, thuat_toan, predict_subjects, result_list):
    training_subjects = get_header_list(khoa=khoa, thuat_toan=thuat_toan)
    if training_subjects == -1:
        return "none_training"
    print("training: ", training_subjects)
    print("predict subject: ", predict_subjects)
    # kiểm tra 2 list có bằng nhau ko
    if training_subjects == predict_subjects:
        return predict_subjects, result_list, [], []
    print("co sai lech")

    print("bắt đầu quá trình chỉnh sửa") 
    # loại bỏ các môn bị thừa
    unecessary_subject, cleaned_subject_list, clean_result_list = Clean_unecessary_subject(predict_subjects=predict_subjects,
                                                                        trainning_subjects=training_subjects,
                                                                        result_list=result_list) 
    
    # thêm các môn bị thiếu
    incompliance_subject, full_subject_list, full_result_list = Add_incompliance_subject_to_predict(
                                                                    predict_subjects=cleaned_subject_list, 
                                                                    trainning_subjects=training_subjects, 
                                                                    result_list=clean_result_list)
    
    # sắp xếp lại thứ tự các môn
    ordered_result_list = Sort_predict_subject_list(predict_subjects=full_subject_list,
                                                    result_list=full_result_list,
                                                    trainning_subjects=training_subjects)


    return training_subjects, ordered_result_list, incompliance_subject, unecessary_subject
    # thêm các môn bị thiếu và chỉnh sửa lại thứ tự                                                                    
    