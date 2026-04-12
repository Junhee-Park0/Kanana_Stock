import warnings

from bs4 import BeautifulSoup, Comment, XMLParsedAsHTMLWarning
import json
import re
from pathlib import Path

from config import Config
SEC_FILE_PATH = Config.SEC_FILE_PATH

class SEC_Parser:
    def __init__(self, ticker : str, file_path : Path):
        self.ticker = ticker
        self.file_path = Path(file_path)

    def parse_filing(self, form : str):
        """[종합] 문서의 종류(form)에 따라 다른 파싱 방법 적용"""
        if form == "4":
            return self.parse_form_4(form)
        elif form == "SC 13G":
            return self.parse_sc_13g(form)
        elif form in ["10-Q", "10-K", "8-K", "DEF 14A"]:
            return self.parse_general_html(form)
        else:
            raise ValueError(f"Unsupported form: {form}")

    def _read_content(self) -> str:
        """여러 인코딩 방법을 통해 파일 읽기"""
        raw = self.file_path.read_bytes()
        for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors = "ignore")

    def _extract_sec_text_block(self, content: str) -> str:
        """문서의 본문 블록 추출"""
        match = re.search(r"<TEXT>(.*?)</TEXT>", content, flags=re.IGNORECASE | re.DOTALL)
        return match.group(1) if match else content

    def _build_soup(self, content: str, parsers):
        """BeautifulSoup 객체 생성"""
        for parser in parsers:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
                    return BeautifulSoup(content, parser)
            except Exception:
                continue
        raise ValueError("No available parser could parse the document.")

    def _to_float(self, value):
        """문자열/숫자 값 정규화하여 float 반환"""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if not text:
            return None

        text = text.replace(",", "")
        text = text.replace("$", "")
        text = re.sub(r"\s+", " ", text)
        number_match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not number_match:
            return None
        try:
            return float(number_match.group(0))
        except ValueError:
            return None

    def _to_bool(self, value) -> bool:
        """문자열 기반 표현을 bool로 변환"""
        if value is None:
            return False
        normalized = str(value).strip().lower()
        return normalized in {"1", "true", "yes", "y"}

    def _tag_local_name(self, tag_name: str) -> str:
        """태그 이름만 소문자로 반환"""
        if not tag_name:
            return ""
        return tag_name.split(":")[-1].lower()

    def _clean_text(self, text: str) -> str:
        """공백 정리"""
        if text is None:
            return ""
        cleaned = text.replace("\xa0", " ")
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)
        cleaned = re.sub(r"\n +", "\n", cleaned)
        return cleaned.strip()

    def _find_first_text_by_local_names(self, soup, local_names):
        """유효한 텍스트 조각 반환"""
        target_names = {name.lower() for name in local_names}
        for tag in soup.find_all(True):
            if self._tag_local_name(tag.name) in target_names:
                text = self._clean_text(tag.get_text(separator = " "))
                if text:
                    return text
        return None

    def _chunk_text(self, text: str, max_chars: int = 3000):
        """문단 기준으로 청킹 (최대 3000자)"""
        if not text:
            return []
        chunks = []
        current = ""

        for paragraph in text.split("\n\n"):
            para = paragraph.strip()
            if not para:
                continue

            if len(para) > max_chars:
                if current:
                    chunks.append(current.strip())
                    current = ""
                for idx in range(0, len(para), max_chars):
                    chunks.append(para[idx: idx + max_chars].strip())
                continue

            candidate = f"{current}\n\n{para}".strip() if current else para
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                current = para

        if current:
            chunks.append(current.strip())

        return chunks

    def parse_form_4(self, form: str):
        """Form 4 XML 파일 파싱"""
        xml_content = self._read_content()
        soup = self._build_soup(xml_content, ("lxml-xml", "xml", "html.parser"))

        def find_ci(tag, name):
            if not tag:
                return None
            return tag.find(lambda node: getattr(node, "name", None) and node.name.lower() == name.lower())

        def find_all_ci(tag, name):
            if not tag:
                return []
            return tag.find_all(lambda node: getattr(node, "name", None) and node.name.lower() == name.lower())

        def get_text(tag, child_tag = None):
            if not tag:
                return None
            target = tag
            if child_tag:
                target = find_ci(tag, child_tag)
            if target:
                # <value> 태그가 있으면 그 안의 텍스트, 없으면 그냥 텍스트
                value_tag = find_ci(target, "value")
                return value_tag.text.strip() if value_tag else target.text.strip()
            return None

        footnotes_map = {
            fn.get("id"): fn.text.strip()
            for fn in find_all_ci(soup, "footnote")
            if fn.get("id")
        }

        def collect_footnotes(target):
            notes = []
            if not target:
                return notes
            for fn_id_tag in find_all_ci(target, "footnoteId"):
                fn_id = fn_id_tag.get("id")
                if fn_id and fn_id in footnotes_map:
                    notes.append(footnotes_map[fn_id])
            return list(dict.fromkeys(notes))

        issuer = find_ci(soup, "issuer")
        owner = find_ci(soup, "reportingOwner")
        owner_relationship = find_ci(owner, "reportingOwnerRelationship") if owner else None

        data = {
            "document_type": form,
            "period_of_report": get_text(soup, "periodOfReport"), # 보고 기준일
            "company_name": get_text(issuer, "issuerName"),
            "ticker": get_text(issuer, "issuerTradingSymbol"),
            "reporter_name": get_text(owner, "rptOwnerName"),
            "reporter_title": get_text(owner_relationship, "officerTitle"), # 직책 (예: CEO, CFO)
            "is_officer": self._to_bool(get_text(owner_relationship, "isOfficer")),
            "is_director": self._to_bool(get_text(owner_relationship, "isDirector")),
            "is_ten_percent": self._to_bool(get_text(owner_relationship, "isTenPercentOwner")),
            "transactions": [] # 거래 내역 리스트
        }

        def build_tx_info(tx, tx_type: str):
            transaction_amounts = find_ci(tx, "transactionAmounts")
            post_amounts = find_ci(tx, "postTransactionAmounts")
            ownership_nature = find_ci(tx, "ownershipNature")
            coding = find_ci(tx, "transactionCoding")
            remarks = collect_footnotes(tx)

            tx_info = {
                "type": tx_type,
                "security_title": get_text(tx, "securityTitle"),
                "date": get_text(tx, "transactionDate"),
                "code": get_text(coding, "transactionCode"), # G(증여), S(매도) 등
                "action": get_text(transaction_amounts, "transactionAcquiredDisposedCode"), # A(취득)/D(처분)
                "shares": self._to_float(get_text(transaction_amounts, "transactionShares")),
                "price": self._to_float(get_text(transaction_amounts, "transactionPricePerShare")),
                "shares_owned_after": self._to_float(get_text(post_amounts, "sharesOwnedFollowingTransaction")),
                "ownership_nature": get_text(ownership_nature, "directOrIndirectOwnership"), # D(직접)/I(간접)
                "remarks": " ".join(remarks) if remarks else ""
            }

            if tx_type == "Derivative":
                tx_info["underlying_security_title"] = get_text(
                    find_ci(tx, "underlyingSecurity"),
                    "underlyingSecurityTitle"
                )
                tx_info["underlying_security_shares"] = self._to_float(
                    get_text(find_ci(tx, "underlyingSecurity"), "underlyingSecurityShares")
                )
                tx_info["exercise_date"] = get_text(tx, "exerciseDate")
                tx_info["expiration_date"] = get_text(tx, "expirationDate")

            return tx_info

        for tx in find_all_ci(soup, "nonDerivativeTransaction"):
            data["transactions"].append(build_tx_info(tx, "Non-Derivative"))

        for tx in find_all_ci(soup, "derivativeTransaction"):
            data["transactions"].append(build_tx_info(tx, "Derivative"))

        file_path = self._save_to_json(data)
        return file_path

    def parse_sc_13g(self, form: str):
        """SC 13G XML/HTML 파일 파싱"""
        content = self._read_content()
        content = self._extract_sec_text_block(content)
        suffix = self.file_path.suffix.lower()

        if suffix == ".xml":
            soup = self._build_soup(content, ("lxml-xml", "xml", "html.parser"))

            def find_ci(tag, name):
                if not tag:
                    return None
                return tag.find(lambda node: getattr(node, "name", None) and node.name.lower() == name.lower())

            def get_val(tag, child_name = None):
                if not tag:
                    return None
                target = tag
                if child_name:
                    target = find_ci(tag, child_name)
                if target:
                    value_tag = find_ci(target, "value")
                    return value_tag.text.strip() if value_tag else target.text.strip()
                return None
            
            issuer = find_ci(soup, "issuer")
            owner = find_ci(soup, "reportingOwner")
            ownership = find_ci(soup, "ownership") or find_ci(soup, "holding")

            data = {
                "document_type": form,
                "company_name": get_val(issuer, "issuerName"),
                "ticker": get_val(issuer, "issuerTradingSymbol"),
                "reporter_name": get_val(owner, "rptOwnerName"),
                "shares_owned": self._to_float(get_val(ownership, "aggregateAmount") or get_val(ownership, "sharesOwned")),
                "percent_of_class": get_val(ownership, "percentOfClass"),
                "is_amendment": "amendment" in content.lower()
            }

            return self._save_to_json(data)

        else:
            soup = self._build_soup(content, ("lxml", "html.parser"))
            for element in soup(["script", "style", "head", "meta", "noscript"]):
                element.decompose()
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()

            raw_text = soup.get_text(separator = "\n").replace("\xa0", " ")
            lines = [
                re.sub(r"\s+", " ", line).strip()
                for line in raw_text.splitlines()
                if line and line.strip()
            ]
            upper_lines = [line.upper() for line in lines]
            compact_text = re.sub(r"\s+", " ", raw_text)

            def collect_values_after(marker: str):
                values = []
                marker = marker.upper()
                ignore_prefixes = ("CHECK BOX", "ROW (9)", "IN ROW", "(A)", "(B)")

                for idx, line in enumerate(upper_lines):
                    if marker not in line:
                        continue
                    for offset in range(1, 8):
                        next_idx = idx + offset
                        if next_idx >= len(lines):
                            break
                        candidate = lines[next_idx]
                        candidate_upper = candidate.upper()
                        if marker in candidate_upper:
                            continue
                        if candidate_upper.startswith(ignore_prefixes):
                            continue
                        if candidate_upper in {"NONE", "NOT APPLICABLE.", "NOT APPLICABLE"}:
                            continue
                        values.append(candidate)
                        break
                return values

            def find_value_before(caption: str):
                caption = caption.upper()
                for idx, line in enumerate(upper_lines):
                    if caption in line and idx > 0:
                        return lines[idx - 1]
                return None

            def normalize_shares(text):
                if not text:
                    return None
                return self._to_float(text)

            def normalize_name(text):
                if not text:
                    return None
                cleaned = re.sub(r"\s+", " ", text).strip(" .;:")
                cleaned = re.sub(r"\s+\d+$", "", cleaned)
                return cleaned

            def normalize_percent(text):
                if not text:
                    return None
                match = re.search(r"\d+(?:\.\d+)?\s*%?", text)
                if not match:
                    return text
                value = match.group(0).replace(" ", "")
                return value if value.endswith("%") else f"{value}%"

            reporter_names = [
                normalize_name(match.group(1))
                for match in re.finditer(
                    r"NAME\s+OF\s+REPORTING\s+PERSONS?\s*(.*?)\s*(?:CHECK\s+THE\s+APPROPRIATE\s+BOX\s+IF\s+A\s+MEMBER\s+OF\s+A\s+GROUP|SEC\s+USE\s+ONLY)",
                    compact_text,
                    flags=re.IGNORECASE
                )
            ]
            shares_values = [
                re.sub(r"\s+", " ", match.group(1)).strip(" .;:")
                for match in re.finditer(
                    r"AGGREGATE\s+AMOUNT\s+BENEFICIALLY\s+OWNED\s+BY\s+EACH\s+REPORTING\s+PERSON\s*(.*?)\s*(?:CHECK\s+BOX\s+IF\s+THE\s+AGGREGATE\s+AMOUNT\s+IN\s+ROW|PERCENT\s+OF\s+CLASS\s+REPRESENTED\s+BY\s+AMOUNT)",
                    compact_text,
                    flags=re.IGNORECASE
                )
            ]
            percent_values = [
                re.sub(r"\s+", " ", match.group(1)).strip(" .;:")
                for match in re.finditer(
                    r"PERCENT\s+OF\s+CLASS\s+REPRESENTED\s+BY\s+AMOUNT\s*(?:IN\s+ROW\s+9)?\s*(.*?)\s*(?:TYPE\s+OF\s+REPORTING\s+PERSON|CUSIP\s+NO\.)",
                    compact_text,
                    flags=re.IGNORECASE
                )
            ]

            if not reporter_names:
                reporter_names = [normalize_name(value) for value in collect_values_after("NAME OF REPORTING PERSON")]
            if not shares_values:
                shares_values = collect_values_after("AGGREGATE AMOUNT BENEFICIALLY OWNED BY EACH REPORTING PERSON")
            if not percent_values:
                percent_values = collect_values_after("PERCENT OF CLASS REPRESENTED BY AMOUNT")

            max_len = max(len(reporter_names), len(shares_values), len(percent_values), 1)
            reporting_persons = []
            seen = set()

            for idx in range(max_len):
                name = reporter_names[idx] if idx < len(reporter_names) else None
                shares = normalize_shares(shares_values[idx]) if idx < len(shares_values) else None
                pct = normalize_percent(percent_values[idx]) if idx < len(percent_values) else None
                key = (name, shares, pct)
                if key in seen:
                    continue
                seen.add(key)
                if any(item is not None for item in key):
                    reporting_persons.append(
                        {
                            "name": name,
                            "shares_owned": shares,
                            "percent_of_class": pct
                        }
                    )

            normalized_text = " ".join(lines)
            amendment_no_match = re.search(r"AMENDMENT NO\.\s*([0-9]+)", normalized_text, flags=re.IGNORECASE)
            is_amendment = bool(re.search(r"\b13G/A\b", normalized_text, flags=re.IGNORECASE))
            is_amendment = is_amendment or bool(amendment_no_match and amendment_no_match.group(1))

            data = {
                "document_type": form,
                "company_name": find_value_before("(NAME OF ISSUER)"),
                "cusip": find_value_before("(CUSIP NUMBER)"),
                "event_date": find_value_before("(DATE OF EVENT WHICH REQUIRES FILING OF THIS STATEMENT)"),
                "reporter_name": reporting_persons[0]["name"] if reporting_persons else None,
                "shares_owned": reporting_persons[0]["shares_owned"] if reporting_persons else None,
                "percent_of_class": reporting_persons[0]["percent_of_class"] if reporting_persons else None,
                "reporting_persons": reporting_persons,
                "is_amendment": is_amendment
            }
            return self._save_to_json(data)

    def parse_general_html(self, form: str):
        """남은 문서들(HTML) 파일 파싱"""
        content = self._read_content()
        content = self._extract_sec_text_block(content)

        is_xbrl_instance = bool(re.search(r"<xbrl[\s>]", content, flags = re.IGNORECASE)) and not bool(
            re.search(r"<html[\s>]", content, flags = re.IGNORECASE)
        )

        if is_xbrl_instance:
            soup = self._build_soup(content, ("lxml-xml", "xml", "html.parser"))

            text_blocks = []
            for tag in soup.find_all(True):
                if self._tag_local_name(tag.name).endswith("textblock"):
                    raw_block = self._clean_text(tag.get_text(separator = "\n"))
                    if "<" in raw_block and ">" in raw_block:
                        html_block = self._build_soup(raw_block, ("lxml", "html.parser"))
                        block_text = self._clean_text(html_block.get_text(separator = "\n"))
                    else:
                        block_text = raw_block
                    if len(block_text) >= 120:
                        text_blocks.append(block_text)

            text_blocks = list(dict.fromkeys(text_blocks))
            normalized_candidates = []
            filtered_blocks = []
            for block in sorted(text_blocks, key = len, reverse = True):
                normalized = re.sub(r"\s+", " ", block).strip().lower()
                if any(normalized in kept for kept in normalized_candidates):
                    continue
                normalized_candidates.append(normalized)
                filtered_blocks.append(block)
            text_blocks = filtered_blocks
            combined_text = "\n\n".join(text_blocks).strip()

            if not combined_text:
                fallback_parts = []
                for tag in soup.find_all(True):
                    local = self._tag_local_name(tag.name)
                    if local in {"context", "unit", "schemaref", "footnote", "identifier"}:
                        continue
                    value = self._clean_text(tag.get_text(separator = " "))
                    if not value:
                        continue
                    if len(value) > 180:
                        fallback_parts.append(value)
                combined_text = "\n\n".join(list(dict.fromkeys(fallback_parts[:200])))

            text_chunks = self._chunk_text(combined_text)
            data = {
                "document_type": form,
                "company_name": self._find_first_text_by_local_names(soup, ["entityregistrantname"]),
                "ticker": self._find_first_text_by_local_names(soup, ["tradingsymbol"]),
                "period_end_date": self._find_first_text_by_local_names(soup, ["documentperiodenddate"]),
                "text_block_count": len(text_blocks),
                "text_chunk_count": len(text_chunks),
                "text_chunks": text_chunks
            }
        else:
            soup = self._build_soup(content, ("lxml", "html.parser"))

            removable_tags = {"script", "style", "head", "title", "meta", "noscript", "ix:header"}
            for element in soup.find_all(lambda tag: tag.name and tag.name.lower() in removable_tags):
                element.decompose()

            for hidden in soup.find_all(
                lambda tag: tag.has_attr("style") and "display:none" in tag.get("style", "").replace(" ", "").lower()
            ):
                hidden.decompose()

            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()

            for td in soup.find_all("td"):
                td.append(" ")
            text = self._clean_text(soup.get_text(separator = "\n"))

            text_chunks = self._chunk_text(text)
            data = {
                "document_type": form,
                "company_name": None,
                "ticker": None,
                "period_end_date": None,
                "text_block_count": 0,
                "text_chunk_count": len(text_chunks),
                "text_chunks": text_chunks
            }

        file_path = self._save_to_json(data)
        return file_path

    def _save_to_json(self, data):
        """파싱 결과를 딕셔러니로 json 파일에 저장 """
        file_name = self.file_path.stem
        base_path = Path(SEC_FILE_PATH) / self.ticker / "Parsed"
        base_path.mkdir(parents = True, exist_ok = True)
        file_path = base_path / f"{file_name}.json"

        with open(file_path, "w", encoding = "utf-8") as f:
            json.dump(data, f, ensure_ascii = False, indent = 4)
        return file_path

if __name__ == "__main__":
    parser = SEC_Parser("AAPL", "data/SEC/AAPL/Raw/aapl-20251227_htm.xml")
    parser.parse_filing("10-K")