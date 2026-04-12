import subprocess
import sys
import shutil
from pathlib import Path

from config import Config


def download_kanana(model_name: str, save_dir: Path) -> None:
    """
    Hugging Face에서 Kanana 모델을 다운로드하는 함수 (최초 1회 실행).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    save_dir.mkdir(parents = True, exist_ok = True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer.save_pretrained(str(save_dir))
    model.save_pretrained(str(save_dir))

    print(f"✅ 모델과 토크나이저가 `{save_dir}`에 저장되었습니다.")


def has_local_kanana(save_dir: Path) -> bool:
    """
    로컬 모델 폴더에 필수 파일이 있는지 확인합니다.
    """
    required_files = [
        save_dir / "config.json",
        save_dir / "tokenizer_config.json",
    ]
    return all(path.exists() for path in required_files)


def ensure_kanana_model() -> None:
    """
    Config에서 model_name, save_dir를 가져와 로컬 모델 존재 여부를 확인하고 필요 시 다운로드합니다.
    """
    model_name = Config.KANANA_MODEL_NAME
    save_dir = Path(Config.KANANA_MODEL_PATH)

    print("Kanana 모델 확인 중..")
    if has_local_kanana(save_dir):
        print("✅ 모델 확인 완료")
        return

    print("로컬 모델이 없어 다운로드를 시작합니다..")
    download_kanana(model_name, save_dir)
    print("✅ 모델 준비 완료")

def ensure_env_file() -> None:
    """
    .env 파일에 www.sec.gov 접근을 위한 USER_EMAIL 설정이 있는지 확인합니다.
    """
    if not Path(".env").exists():
        raise FileNotFoundError(".env 파일이 없습니다. 환경 변수를 설정해주세요.")
    
    with open(".env", "r") as f:
        for line in f:
            if "USER_EMAIL" in line:
                return
    
    raise ValueError("USER_EMAIL이 설정되어 있지 않습니다. .env 파일에 USER_EMAIL을 설정해주세요.")


def _run_cmd(cmd: list[str]) -> None:
    """명령 실행 헬퍼 (실패 시 예외 발생)."""
    subprocess.run(cmd, check = True)


def ensure_linux_selenium_runtime() -> None:
    """
    Linux 환경의 경우, Selenium 실행에 필요한 브라우저/라이브러리/패키지를 설치합니다.
    """
    if not sys.platform.startswith("linux"):
        print("Linux 환경이 아니므로 Selenium 런타임 설치를 건너뜁니다.")
        return

    print("Linux Selenium 런타임 점검을 시작합니다.")

    if not shutil.which("apt-get"):
        print("apt-get이 없어 시스템 패키지 설치를 건너뜁니다.")
    else:
        _run_cmd(["apt-get", "update"])

        # Chrome/Chromium 런타임 의존 라이브러리 설치 (요청 반영)
        try:
            _run_cmd([
                "apt-get", "install", "-y",
                "libatk1.0-0", "libatk-bridge2.0-0", "libcups2", "libdrm2", "libxkbcommon0",
                "libxcomposite1", "libxdamage1", "libxfixes3", "libxrandr2", "libgbm1", "libgtk-3-0",
                "libasound2t64",
            ])
        except subprocess.CalledProcessError:
            _run_cmd(["apt-get", "install", "-y", "libasound2"])

        try:
            # Ubuntu/Debian 계열: chromium-browser가 있는 경우
            _run_cmd(["apt-get", "install", "-y", "chromium-browser", "libnss3", "libnspr4"])
        except subprocess.CalledProcessError:
            # 일부 배포판은 chromium 패키지명만 제공합니다.
            _run_cmd(["apt-get", "install", "-y", "chromium", "libnss3", "libnspr4"])

    # Selenium + 드라이버 매니저 파이썬 패키지 설치
    _run_cmd([sys.executable, "-m", "pip", "install", "-U", "selenium", "webdriver-manager"])
    print("✅ Linux Selenium 런타임 설치 완료")

def main() -> None:
    print("초기 환경 점검을 시작합니다.")
    ensure_linux_selenium_runtime()
    ensure_kanana_model()
    print("모든 초기 설정이 완료되었습니다.")


if __name__ == "__main__":
    main()