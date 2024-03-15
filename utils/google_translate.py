from google.cloud import translate
import os
# credential_path = "utils\M400806-7257e31dba3d.json"
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_path
# Initialize Translation client
def translate_text(
    text: str = "YOUR_TEXT_TO_TRANSLATE", project_id: str = "agent-400806"
) -> translate.TranslationServiceClient:
    """Translating Text."""

    client = translate.TranslationServiceClient()

    location = "global"

    parent = f"projects/{project_id}/locations/{location}"

    # Translate text from English to French
    # Detail on supported types can be found here:
    # https://cloud.google.com/translate/docs/supported-formats
    if not text:
        text = "Oh, No"
    response = client.translate_text(
        request={
            "parent": parent,
            "contents": [text],
            "mime_type": "text/plain",  # mime types: text/plain, text/html
            "source_language_code": "zh-CN",
            "target_language_code": "en-US",
        }
    )

    return response.translations[0].translated_text


if __name__ == "__main__":
    res = translate_text(
        "你的任务是判断输入的医疗报告的类型，报告类型可选项：['其他', '检验记录', '医嘱单', '处方单', '注射单', '费用单', '体检报告', '基因检测', '手术记录', '检查记录', '治疗记录','病理报告', '病程记录', '门诊病历', '出入院记录', '知情同意书', '其他会诊记录', '病理会诊记录', '其他疾病诊断书', '门诊疾病诊断书', '出入院疾病诊断书'] \n医疗报告：{content} \n输出格式：直接从可选项中选择最恰当的一项并输出，无需解释或输出多余内容"
    )
    print(res)
