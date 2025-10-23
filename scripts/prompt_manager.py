"""
Kaggle VQA Challenge - Prompt Manager

This script manages prompt templates and builds messages for Qwen2.5-VL
using apply_chat_template and process_vision_info for compatibility.
"""

import yaml
from pathlib import Path
from typing import Dict, List


class PromptManager:
    """
    프롬프트 템플릿 관리자

    ✅ CRITICAL: Uses apply_chat_template + process_vision_info
                 for Qwen2.5-VL compatibility
    """

    def __init__(self, templates_path: str = 'config/prompt_templates.yaml'):
        """
        Args:
            templates_path: 프롬프트 템플릿 YAML 파일 경로
        """
        self.templates_path = templates_path
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict:
        """템플릿 파일 로드"""
        templates_file = Path(self.templates_path)

        if not templates_file.exists():
            raise FileNotFoundError(
                f"Template file not found: {self.templates_path}\n"
                f"Please run: python scripts/prompt_manager.py --create_template"
            )

        with open(templates_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if 'prompt_templates' not in config:
            raise ValueError(
                f"Invalid template file: Missing 'prompt_templates' key"
            )

        return config['prompt_templates']

    def format_prompt(
        self,
        question_type: str,
        question: str,
        choices: Dict[str, str]
    ) -> Dict[str, str]:
        """
        질문 유형에 맞는 프롬프트 생성

        Args:
            question_type: 'counting', 'color', 'ocr', etc.
            question: 질문 텍스트
            choices: {'a': '...', 'b': '...', 'c': '...', 'd': '...'}

        Returns:
            dict: {
                'system': str,
                'user': str
            }
        """
        # 템플릿 선택 (없으면 general 사용)
        template = self.templates.get(question_type, self.templates.get('general'))

        if template is None:
            raise ValueError(f"No template found for type: {question_type}")

        # 시스템 프롬프트
        system_prompt = template.get('system', '')

        # 사용자 프롬프트 (변수 치환)
        user_template = template.get('user', '')
        user_prompt = user_template.format(
            question=question,
            choice_a=choices.get('a', ''),
            choice_b=choices.get('b', ''),
            choice_c=choices.get('c', ''),
            choice_d=choices.get('d', '')
        )

        return {
            'system': system_prompt,
            'user': user_prompt
        }

    def build_messages(
        self,
        image_path: str,
        question_type: str,
        question: str,
        choices: Dict[str, str]
    ) -> List[Dict]:
        """
        ✅ CRITICAL: Qwen2.5-VL 표준 메시지 형식 생성

        이 메서지는 apply_chat_template + process_vision_info와 함께 사용됨

        Args:
            image_path: 이미지 파일 경로
            question_type: 질문 유형
            question: 질문 텍스트
            choices: {'a': '...', 'b': '...', 'c': '...', 'd': '...'}

        Returns:
            list: Qwen2.5-VL 호환 메시지 형식
                [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "..."}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": "..."},
                            {"type": "text", "text": "..."}
                        ]
                    }
                ]

        Usage:
            messages = prompt_manager.build_messages(...)
            text = processor.apply_chat_template(messages, tokenize=False)
            images, videos = process_vision_info(messages)
            inputs = processor(text=[text], images=images, videos=videos)
        """
        # 프롬프트 포맷팅
        prompt = self.format_prompt(question_type, question, choices)

        # Qwen2.5-VL 표준 메시지 구조
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": prompt['system']}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt['user']}
                ]
            }
        ]

        return messages

    def build_training_messages(
        self,
        image_path: str,
        question_type: str,
        question: str,
        choices: Dict[str, str],
        answer: str
    ) -> List[Dict]:
        """
        ✅ CRITICAL: 학습용 메시지 생성 (assistant 응답 포함)

        라벨 정렬을 위해 assistant 메시지에 정답 1글자 포함

        Args:
            image_path: 이미지 파일 경로
            question_type: 질문 유형
            question: 질문 텍스트
            choices: {'a': '...', 'b': '...', 'c': '...', 'd': '...'}
            answer: 정답 ('a', 'b', 'c', 'd')

        Returns:
            list: 학습용 메시지 (assistant 응답 포함)
                [
                    {"role": "system", "content": [...]},
                    {"role": "user", "content": [...]},
                    {"role": "assistant", "content": [{"type": "text", "text": "a"}]}
                ]

        Usage:
            messages = prompt_manager.build_training_messages(...)
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False  # ✅ False for training!
            )
        """
        # 기본 메시지 (system + user)
        messages = self.build_messages(image_path, question_type, question, choices)

        # ✅ CRITICAL: assistant 응답 추가 (정답 1글자)
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": answer.lower()}]  # a, b, c, d
        })

        return messages

    def get_available_types(self) -> List[str]:
        """사용 가능한 질문 유형 목록 반환"""
        return list(self.templates.keys())

    def print_template(self, question_type: str):
        """특정 유형의 템플릿 출력"""
        if question_type not in self.templates:
            print(f"❌ No template found for type: {question_type}")
            print(f"Available types: {self.get_available_types()}")
            return

        template = self.templates[question_type]
        print(f"\n{'='*60}")
        print(f"Template: {question_type}")
        print(f"{'='*60}\n")
        print(f"System:\n{template['system']}\n")
        print(f"User:\n{template['user']}")

    def test_prompt(
        self,
        question_type: str = 'general',
        question: str = "이미지에 사과가 몇 개 있습니까?",
        choices: Dict[str, str] = None
    ):
        """프롬프트 테스트"""
        if choices is None:
            choices = {
                'a': '1개',
                'b': '2개',
                'c': '3개',
                'd': '4개'
            }

        print(f"\n{'='*60}")
        print(f"Testing Prompt: {question_type}")
        print(f"{'='*60}\n")

        prompt = self.format_prompt(question_type, question, choices)

        print(f"System Prompt:\n{'-'*60}")
        print(prompt['system'])
        print(f"\n{'─'*60}")
        print(f"User Prompt:\n{'-'*60}")
        print(prompt['user'])

        print(f"\n{'='*60}")


def create_default_template(output_path: str = 'config/prompt_templates.yaml'):
    """기본 템플릿 파일 생성 (독립 실행용)"""
    from pathlib import Path

    # 이미 존재하면 경고
    if Path(output_path).exists():
        print(f"⚠️  Template file already exists: {output_path}")
        return

    # 템플릿 내용은 이미 config/prompt_templates.yaml에 작성됨
    print(f"✓ Template file should be created at: {output_path}")
    print(f"  Please ensure the file exists with proper content.")


def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='Prompt Manager for VQA')
    parser.add_argument('--templates', default='config/prompt_templates.yaml', help='Template file path')
    parser.add_argument('--list_types', action='store_true', help='List available question types')
    parser.add_argument('--show_template', type=str, help='Show template for specific type')
    parser.add_argument('--test_prompt', type=str, help='Test prompt for specific type')

    args = parser.parse_args()

    try:
        manager = PromptManager(templates_path=args.templates)

        if args.list_types:
            print("\nAvailable Question Types:")
            for qtype in manager.get_available_types():
                print(f"  - {qtype}")

        elif args.show_template:
            manager.print_template(args.show_template)

        elif args.test_prompt:
            manager.test_prompt(question_type=args.test_prompt)

        else:
            print("Prompt Manager initialized successfully!")
            print(f"Templates loaded from: {args.templates}")
            print(f"Available types: {manager.get_available_types()}")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nTo create default template:")
        print("  Ensure config/prompt_templates.yaml exists")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
