from csv import reader
import torch


# Read the data from the file and return dict
def readData():
    id2data = {}
    titles = list

    # open file, iterate over rows
    with open('diab_ckd_data.csv', 'r') as dataFile:
        csv_reader = reader(dataFile)
        for i, row in enumerate(csv_reader):
            # first line - save titles
            if i == 0:
                titles = row
            # next lines - save in data dict
            else:
                ID = int(row[0])
                patient = {}
                for title, value in zip(titles[1:], row[1:]):
                    patient[title] = value
                id2data[ID] = patient

    return id2data


# Receive data dict and return it encoded
def encodeData(dataDict):
    # One-hot encode the age group
    def ecodeAgeGroup(ageGroupStr):
        if ageGroupStr == "[18, 30)":
            return [1, 0, 0, 0, 0]
        elif ageGroupStr == "[30, 45)":
            return [0, 1, 0, 0, 0]
        elif ageGroupStr == "[45, 60)":
            return [0, 0, 1, 0, 0]
        elif ageGroupStr == "[60, 75)":
            return [0, 0, 0, 1, 0]
        elif ageGroupStr == "[75, 120)":
            return [0, 0, 0, 0, 1]

    # One-hot encode the SES group
    def encodeSES_GROUP(SES_GROUPStr):
        if SES_GROUPStr == "LOW":
            return [1, 0, 0, 0]
        elif SES_GROUPStr == "MID":
            return [0, 1, 0, 0]
        elif SES_GROUPStr == "HI":
            return [0, 0, 1, 0]
        elif SES_GROUPStr == "OTHER":
            return [0, 0, 0, 1]

    # One-hot encode the migzar
    def encodeMIGZAR(MIGZARStr):
        if MIGZARStr == "ARAB":
            return [1, 0, 0]
        elif MIGZARStr == "GENERAL":
            return [0, 1, 0]
        elif MIGZARStr == "HAREDI":
            return [0, 0, 1]

    # value + 1/-1 if the value is NA
    def encodeRegressiveOrNA(valueStr):
        if valueStr != "NA":
            return [float(valueStr), 1]
        elif valueStr == "NA":
            return [0, -1]

    # 1/-1 instead of 1/0
    def encodeBinary(valueStr):
        return [1] if valueStr == '1' else [-1]

    # for each patient: encode all data fields, concatenate, and save it in the dict as tensor
    id2encodedData = {}
    for ID, patientData in dataDict.items():
        encoded_TIME_CRF = [float(patientData["TIME_CRF"])]
        encoded_EVENT_CRF = [int(patientData["EVENT_CRF"])]
        encoded_IS_MALE = encodeBinary(patientData["IS_MALE"])
        encoded_AGE_AT_SDATE = [float(patientData["AGE_AT_SDATE"])]
        encoded_AGE_GROUP = ecodeAgeGroup(patientData["AGE_GROUP"])
        encoded_SES_GROUP = encodeSES_GROUP(patientData["SES_GROUP"])
        encoded_MIGZAR = encodeMIGZAR(patientData["MIGZAR"])
        encoded_IS_HYPERTENSION = encodeBinary(patientData["IS_HYPERTENSION"])
        encoded_SE_HYPERTENSION = encodeRegressiveOrNA(patientData["SE_HYPERTENSION"])
        encoded_IS_ISCHEMIC_MI = encodeBinary(patientData["IS_ISCHEMIC_MI"])
        encoded_SE_ISCHEMIC_MI = encodeRegressiveOrNA(patientData["SE_ISCHEMIC_MI"])
        encoded_IS_CVA_TIA = encodeBinary(patientData["IS_CVA_TIA"])
        encoded_SE_CVA_TIA = encodeRegressiveOrNA(patientData["SE_CVA_TIA"])
        encoded_IS_DEMENTIA = encodeBinary(patientData["IS_DEMENTIA"])
        encoded_SE_DEMENTIA = encodeRegressiveOrNA(patientData["SE_DEMENTIA"])
        encoded_IS_ART_SCLE_GEN = encodeBinary(patientData["IS_ART_SCLE_GEN"])
        encoded_SE_ART_SCLE_GEN = encodeRegressiveOrNA(patientData["SE_ART_SCLE_GEN"])
        encoded_IS_TROMBOPHILIA = encodeBinary(patientData["IS_TROMBOPHILIA"])
        encoded_SE_TROMBOPHILIA = encodeRegressiveOrNA(patientData["SE_TROMBOPHILIA"])
        encoded_IS_IBD = encodeBinary(patientData["IS_IBD"])
        encoded_SE_IBD = encodeRegressiveOrNA(patientData["SE_IBD"])
        encoded_BMI_AT_BASELINE = encodeRegressiveOrNA(patientData["BMI_AT_BASELINE"])
        encoded_SYSTOLA_AT_BASELINE = encodeRegressiveOrNA(patientData["SYSTOLA_AT_BASELINE"])
        encoded_DIASTOLA_AT_BASELINE = encodeRegressiveOrNA(patientData["DIASTOLA_AT_BASELINE"])
        encoded_Creatinine_B_AT_BASELINE = encodeRegressiveOrNA(patientData["Creatinine_B_AT_BASELINE"])
        encoded_Albumin_B_AT_BASELINE = encodeRegressiveOrNA(patientData["Albumin_B_AT_BASELINE"])
        encoded_Urea_B_AT_BASELINE = encodeRegressiveOrNA(patientData["Urea_B_AT_BASELINE"])
        encoded_Glucose_B_AT_BASELINE = encodeRegressiveOrNA(patientData["Glucose_B_AT_BASELINE"])
        encoded_HbA1C_AT_BASELINE = encodeRegressiveOrNA(patientData["HbA1C_AT_BASELINE"])
        encoded_RBCRed_Blood_Cells_AT_BASELINE = encodeRegressiveOrNA(patientData["RBCRed_Blood_Cells_AT_BASELINE"])
        encoded_Hemoglobin_AT_BASELINE = encodeRegressiveOrNA(patientData["Hemoglobin_AT_BASELINE"])
        encoded_Ferritin_AT_BASELINE = encodeRegressiveOrNA(patientData["Ferritin_AT_BASELINE"])
        encoded_AST_GOT_AT_BASELINE = encodeRegressiveOrNA(patientData["AST_GOT_AT_BASELINE"])
        encoded_ALT_GPT_AT_BASELINE = encodeRegressiveOrNA(patientData["ALT_GPT_AT_BASELINE"])
        encoded_Bilirubin_Total_AT_BASELINE = encodeRegressiveOrNA(patientData["Bilirubin_Total_AT_BASELINE"])
        encoded_Na_Sodium_B_AT_BASELINE = encodeRegressiveOrNA(patientData["Na_Sodium_B_AT_BASELINE"])
        encoded_K_Potassium_B_AT_BASELINE = encodeRegressiveOrNA(patientData["K_Potassium_B_AT_BASELINE"])
        encoded_CaCalcium_B_AT_BASELINE = encodeRegressiveOrNA(patientData["CaCalcium_B_AT_BASELINE"])
        encoded_HDLCholesterol_AT_BASELINE = encodeRegressiveOrNA(patientData["HDLCholesterol_AT_BASELINE"])
        encoded_LDLCholesterol_AT_BASELINE = encodeRegressiveOrNA(patientData["LDLCholesterol_AT_BASELINE"])
        encoded_Triglycerides_AT_BASELINE = encodeRegressiveOrNA(patientData["Triglycerides_AT_BASELINE"])
        encoded_PTH_AT_BASELINE = encodeRegressiveOrNA(patientData["PTH_AT_BASELINE"])
        encoded_all = encoded_TIME_CRF + encoded_EVENT_CRF + encoded_IS_MALE + encoded_AGE_AT_SDATE + encoded_AGE_GROUP + encoded_SES_GROUP + encoded_MIGZAR + encoded_IS_HYPERTENSION + encoded_SE_HYPERTENSION + \
                      encoded_IS_ISCHEMIC_MI + encoded_SE_ISCHEMIC_MI + encoded_IS_CVA_TIA + encoded_SE_CVA_TIA + encoded_IS_DEMENTIA + encoded_SE_DEMENTIA + encoded_IS_ART_SCLE_GEN + encoded_SE_ART_SCLE_GEN + \
                      encoded_IS_TROMBOPHILIA + encoded_SE_TROMBOPHILIA + encoded_IS_IBD + encoded_SE_IBD + encoded_BMI_AT_BASELINE + encoded_SYSTOLA_AT_BASELINE + encoded_DIASTOLA_AT_BASELINE + encoded_Creatinine_B_AT_BASELINE + \
                      encoded_Albumin_B_AT_BASELINE + encoded_Urea_B_AT_BASELINE + encoded_Glucose_B_AT_BASELINE + encoded_HbA1C_AT_BASELINE + encoded_RBCRed_Blood_Cells_AT_BASELINE + encoded_Hemoglobin_AT_BASELINE + \
                      encoded_Ferritin_AT_BASELINE + encoded_AST_GOT_AT_BASELINE + encoded_ALT_GPT_AT_BASELINE + encoded_Bilirubin_Total_AT_BASELINE + encoded_Na_Sodium_B_AT_BASELINE + encoded_K_Potassium_B_AT_BASELINE + \
                      encoded_CaCalcium_B_AT_BASELINE + encoded_HDLCholesterol_AT_BASELINE + encoded_LDLCholesterol_AT_BASELINE + encoded_Triglycerides_AT_BASELINE + encoded_PTH_AT_BASELINE
        id2encodedData[ID] = torch.Tensor(encoded_all)
    return id2encodedData


if __name__ == '__main__':
    dataRaw = readData()
    dataEncoded = encodeData(dataRaw)

