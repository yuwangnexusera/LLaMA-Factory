import tiktoken


class TokenCalculate:
    def __init__(self, model) -> None:
        self.model = model

    def token_count(self, string):
        # TODO baichuan
        try:
            encoding = tiktoken.encoding_for_model(self.model)  # gpt-4/gpt-3.5-turbo
        except:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        token_num = len(encoding.encode(string))
        return token_num


if __name__ == "__main__":
    s1 = """Your task is to extract medical information from the input report and output it in JSON format{
            "Basic Information": ["Date of Birth", "Age", "Gender"],
            "Disease": [
                "Date of First Diagnosis",
                "Time of First Pathological Diagnosis (Biopsy, Post-operative Pathology, etc.)",
                "Time of First Lung Resection",
                "Time of First Imaging Diagnosis",
                "Time of First Treatment (Drugs, Radiotherapy, etc.)",
                "Time of First Symptom",
                "Disease Name",
            ],
            "Symptom": ["ECOG Score", "ECOG Date"],
            "Diagnosis": ["Diagnosing Doctor"],
            "Imaging": ["Brain Metastasis Date", "Brain Metastasis Site"],
            "Pathology": ["Pathology Date", "Pathology Type"],
            "Genetic Testing": [
                "ALK",
                "MET",
                "RB1",
                "RET",
                "BRAF",
                "BRCA",
                "EGFR",
                "FGFR",
                "KRAS",
                "NTRK",
                "ROS1",
                "TP53",
                "KEAP1",
                "STK11",
                "HER2 (ERBB2)",
                "HER3 (ERBB3)",
                "HER4 (ERBB4)",
                "Genetic Testing Date",
            ],
            "Immune Testing": [
                "Immune Cell",
                "Combined Positive Score",
                "Tumor Proportion Score",
                "PD-L1",
                "Immunological Test Date",
            ],
            "Cancer treatment": [
                "Surgical Site",
                "Treatment Start Date",
                "Treatment Drug Names",
                "Treatment End Date",
                "Specific Tumor Treatment Method",
            ],
            "Treatment Drug Plan": [
                "Treatment Start Date",
                "Treatment Drug Names",
                "Treatment End Date",
                "Is Treatment Drug Recommended",
            ],
            "Comorbid Disease": [
                "Date of Confirmed Disease",
                "Information Source",
                "Infectious Diseases",
                "Respiratory System Diseases",
                "Circulatory System Diseases",
                "Malignant Tumor Conditions",
                "Digestive System Diseases",
                "Nervous System Diseases",
                "Urogenital System Diseases",
                "Eye, Ear, Nose, and Throat Related Diseases",
                "Endocrine and Immune System Diseases",
            ],
            "Date": ["Admission Date", "Discharge Date", "Medical History Collection Date", "Record Date"],
        }. Input report:Name of discharged patient: [patient's name]\nGender: female\nAge: [specific age]\n\nDate of discharge: 2022-05-01\nDate of admission: 2022-04-27\nDate of medical history collection: 2024-11-10\n\nI. Basic information\nThe patient was first diagnosed with intestinal adenocarcinoma on 2019-06-15, and the diagnosis was confirmed by Zhao Siming. During this period, the patient had circulatory and endocrine immune system diseases such as thrombosis and systemic lupus erythematosus. The patient received cell therapy, starting on 2020-06-03 and ending on 2020-06-04. In addition, the patient had a history of infectious diseases such as tuberculosis, and the first imaging diagnosis was confirmed on 2021-04-11. The ECOG score was 3, and the evaluation date was 2023-01-27.\n\n2. Pathological diagnosis\nThe pathological types of lung cancer in patients include adenocarcinoma in situ and intestinal adenocarcinoma, and the genes involved include KEAP1 positive, HER4 (ERBB4) positive, EGFR mutation (C797S, L861Q), BRCA positive, and other rare mutations of KRAS. The date of immune detection is 2019-07-06, and the date of genetic detection is 2024-06-09. No brain metastasis was found this time.\n\n3. Surgery\nThe patient underwent lung surgery, and the specific surgical site was lung resection.\n\n4. Treatment and medication\nDuring this treatment, the patient used drugs for lung cancer, and the specific drug name is [Lung cancer treatment drugs].\n\n5. Discharge instructions\nIt is recommended that patients continue to use drugs according to doctor's instructions after discharge, and return to the hospital for follow-up examinations regularly. Pay attention to rest, strengthen nutrition, follow a healthy lifestyle, and regularly monitor physical condition to detect possible recurrence or metastasis as soon as possible.\n\n6. Follow-up plan\nAfter discharge, patients need to be followed up regularly, including imaging examinations, hematological examinations, and necessary genetic tests to monitor the progression of the disease.\n\nVII. Doctors and medical team\nDiagnostic doctor: [Zhao Si's name]\nAttending doctor: [Attending doctor's name]\nNursing team: [List of nursing team members]\n\nNote: Specific information needs to be filled in according to the actual situation of the patient. JSON result:"""
    print(TokenCalculate("gpt-3.5-turbo").token_count(s1))
