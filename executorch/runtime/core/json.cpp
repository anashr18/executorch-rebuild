#include "runtime/core/json.h"

#include <cctype>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace executorch::runtime::core {

namespace {
std::string indent_string(int indent) {
  return std::string(static_cast<std::size_t>(indent), ' ');
}
} // namespace

JsonValue::JsonValue() noexcept : value_(nullptr) {}

JsonValue::JsonValue(value_type value) : value_(std::move(value)) {}

bool JsonValue::is_null() const noexcept {
  return std::holds_alternative<std::nullptr_t>(value_);
}

bool JsonValue::is_bool() const noexcept {
  return std::holds_alternative<bool>(value_);
}

bool JsonValue::is_number() const noexcept {
  return std::holds_alternative<double>(value_);
}

bool JsonValue::is_string() const noexcept {
  return std::holds_alternative<std::string>(value_);
}

bool JsonValue::is_array() const noexcept {
  return std::holds_alternative<array_type>(value_);
}

bool JsonValue::is_object() const noexcept {
  return std::holds_alternative<object_type>(value_);
}

bool JsonValue::as_bool() const {
  if (!is_bool()) {
    throw std::runtime_error("JsonValue: expected boolean");
  }
  return std::get<bool>(value_);
}

double JsonValue::as_number() const {
  if (!is_number()) {
    throw std::runtime_error("JsonValue: expected number");
  }
  return std::get<double>(value_);
}

const std::string &JsonValue::as_string() const {
  if (!is_string()) {
    throw std::runtime_error("JsonValue: expected string");
  }
  return std::get<std::string>(value_);
}

const JsonValue::array_type &JsonValue::as_array() const {
  if (!is_array()) {
    throw std::runtime_error("JsonValue: expected array");
  }
  return std::get<array_type>(value_);
}

const JsonValue::object_type &JsonValue::as_object() const {
  if (!is_object()) {
    throw std::runtime_error("JsonValue: expected object");
  }
  return std::get<object_type>(value_);
}

const JsonValue &JsonValue::at(const std::string &key) const {
  const auto &obj = as_object();
  const auto it = obj.find(key);
  if (it == obj.end()) {
    throw std::runtime_error("JsonValue: missing key '" + key + "'");
  }
  return it->second;
}

std::string JsonValue::dump(int indent) const {
  std::ostringstream out;
  if (is_null()) {
    out << "null";
  } else if (is_bool()) {
    out << (as_bool() ? "true" : "false");
  } else if (is_number()) {
    out << as_number();
  } else if (is_string()) {
    std::ostringstream escaped;
    escaped << '"';
    for (char ch : as_string()) {
      switch (ch) {
      case '\\':
      case '"':
        escaped << '\\' << ch;
        break;
      case '\b':
        escaped << "\\b";
        break;
      case '\f':
        escaped << "\\f";
        break;
      case '\n':
        escaped << "\\n";
        break;
      case '\r':
        escaped << "\\r";
        break;
      case '\t':
        escaped << "\\t";
        break;
      default:
        escaped << ch;
      }
    }
    escaped << '"';
    out << escaped.str();
  } else if (is_array()) {
    out << "[\n";
    const auto &arr = as_array();
    for (std::size_t i = 0; i < arr.size(); ++i) {
      out << indent_string(indent + 2) << arr[i].dump(indent + 2);
      if (i + 1 < arr.size()) {
        out << ",";
      }
      out << "\n";
    }
    out << indent_string(indent) << "]";
  } else if (is_object()) {
    out << "{\n";
    const auto &obj = as_object();
    std::size_t index = 0;
    for (const auto &[key, value] : obj) {
      out << indent_string(indent + 2) << '"' << key
          << "\": " << value.dump(indent + 2);
      if (index + 1 < obj.size()) {
        out << ",";
      }
      out << "\n";
      ++index;
    }
    out << indent_string(indent) << "}";
  }
  return out.str();
}

JsonParser::JsonParser(std::string source)
    : source_(std::move(source)), data_(source_.data()), size_(source_.size()),
      pos_(0) {}

JsonValue JsonParser::parse() {
  skip_whitespace();
  JsonValue value = parse_value();
  skip_whitespace();
  if (pos_ != size_) {
    raise("Trailing characters after JSON payload");
  }
  return value;
}

void JsonParser::skip_whitespace() {
  while (pos_ < size_ &&
         std::isspace(static_cast<unsigned char>(data_[pos_]))) {
    ++pos_;
  }
}

char JsonParser::peek() const {
  if (pos_ >= size_) {
    throw std::runtime_error("Unexpected end of JSON input");
  }
  return data_[pos_];
}

char JsonParser::consume() {
  const char ch = peek();
  ++pos_;
  return ch;
}

bool JsonParser::consume_if(char expected) {
  if (pos_ < size_ && data_[pos_] == expected) {
    ++pos_;
    return true;
  }
  return false;
}

JsonValue JsonParser::parse_value() {
  if (pos_ >= size_) {
    raise("Unexpected end of JSON input when parsing value");
  }
  const char ch = peek();
  switch (ch) {
  case '{':
    return parse_object();
  case '[':
    return parse_array();
  case '"':
    return parse_string();
  case 't':
    return parse_literal("true", JsonValue{JsonValue::value_type{true}});
  case 'f':
    return parse_literal("false", JsonValue{JsonValue::value_type{false}});
  case 'n':
    return parse_literal("null", JsonValue{JsonValue::value_type{nullptr}});
  default:
    if (ch == '-' || std::isdigit(static_cast<unsigned char>(ch))) {
      return parse_number();
    }
    raise(std::string("Unexpected character '") + ch +
          "' while parsing JSON value");
  }
}

JsonValue JsonParser::parse_object() {
  consume(); // '{'
  JsonValue::object_type object;
  skip_whitespace();
  if (consume_if('}')) {
    return JsonValue(object);
  }
  while (true) {
    skip_whitespace();
    JsonValue key = parse_string();
    skip_whitespace();
    if (!consume_if(':')) {
      raise("Expected ':' after object key");
    }
    skip_whitespace();
    JsonValue value = parse_value();
    object.emplace(key.as_string(), std::move(value));
    skip_whitespace();
    if (consume_if('}')) {
      break;
    }
    if (!consume_if(',')) {
      raise("Expected ',' between object members");
    }
    skip_whitespace();
  }
  return JsonValue(object);
}

JsonValue JsonParser::parse_array() {
  consume(); // '['
  JsonValue::array_type array;
  skip_whitespace();
  if (consume_if(']')) {
    return JsonValue(array);
  }
  while (true) {
    skip_whitespace();
    array.emplace_back(parse_value());
    skip_whitespace();
    if (consume_if(']')) {
      break;
    }
    if (!consume_if(',')) {
      raise("Expected ',' between array elements");
    }
  }
  return JsonValue(array);
}

JsonValue JsonParser::parse_number() {
  const std::size_t start = pos_;
  if (consume_if('-')) {
  }
  while (pos_ < size_ &&
         std::isdigit(static_cast<unsigned char>(data_[pos_]))) {
    ++pos_;
  }
  if (consume_if('.')) {
    if (pos_ >= size_ ||
        !std::isdigit(static_cast<unsigned char>(data_[pos_]))) {
      raise("Invalid fractional part in number");
    }
    while (pos_ < size_ &&
           std::isdigit(static_cast<unsigned char>(data_[pos_]))) {
      ++pos_;
    }
  }
  if (pos_ < size_ && (data_[pos_] == 'e' || data_[pos_] == 'E')) {
    ++pos_;
    if (pos_ < size_ && (data_[pos_] == '+' || data_[pos_] == '-')) {
      ++pos_;
    }
    if (pos_ >= size_ ||
        !std::isdigit(static_cast<unsigned char>(data_[pos_]))) {
      raise("Invalid exponent in number");
    }
    while (pos_ < size_ &&
           std::isdigit(static_cast<unsigned char>(data_[pos_]))) {
      ++pos_;
    }
  }
  const std::string token = source_.substr(start, pos_ - start);
  char *end_ptr = nullptr;
  const double value = std::strtod(token.c_str(), &end_ptr);
  if (end_ptr != token.c_str() + token.size()) {
    raise("Failed to parse number token '" + token + "'");
  }
  return JsonValue(JsonValue::value_type{value});
}

JsonValue JsonParser::parse_string() {
  if (!consume_if('"')) {
    raise("Expected '\"' to start string");
  }
  std::string result;
  while (pos_ < size_) {
    const char ch = consume();
    if (ch == '"') {
      return JsonValue(JsonValue::value_type{std::move(result)});
    }
    if (ch == '\\') {
      if (pos_ >= size_) {
        raise("Unterminated escape sequence in string");
      }
      const char esc = consume();
      switch (esc) {
      case '"':
      case '\\':
      case '/':
        result.push_back(esc);
        break;
      case 'b':
        result.push_back('\b');
        break;
      case 'f':
        result.push_back('\f');
        break;
      case 'n':
        result.push_back('\n');
        break;
      case 'r':
        result.push_back('\r');
        break;
      case 't':
        result.push_back('\t');
        break;
      default:
        raise(std::string("Unsupported escape sequence \\") + esc);
      }
      continue;
    }
    result.push_back(ch);
  }
  raise("Unterminated string literal");
}

JsonValue JsonParser::parse_literal(const char *literal, JsonValue value) {
  for (const char *cursor = literal; *cursor != '\0'; ++cursor) {
    if (pos_ >= size_ || data_[pos_] != *cursor) {
      raise(std::string("Invalid literal, expected '") + literal + "'");
    }
    ++pos_;
  }
  return JsonValue(std::move(value));
}

[[noreturn]] void JsonParser::raise(const std::string &message) const {
  throw std::runtime_error("JSON parse error at position " +
                           std::to_string(pos_) + ": " + message);
}

JsonValue parse_json(std::string source) {
  JsonParser parser{std::move(source)};
  return parser.parse();
}

} // namespace executorch::runtime::core
