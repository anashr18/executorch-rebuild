#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace executorch::runtime::core {

class JsonValue {
 public:
  using array_type = std::vector<JsonValue>;
  using object_type = std::map<std::string, JsonValue>;
  using value_type =
      std::variant<std::nullptr_t, bool, double, std::string, array_type, object_type>;

  JsonValue() noexcept;
  explicit JsonValue(value_type value);

  bool is_null() const noexcept;
  bool is_bool() const noexcept;
  bool is_number() const noexcept;
  bool is_string() const noexcept;
  bool is_array() const noexcept;
  bool is_object() const noexcept;

  bool as_bool() const;
  double as_number() const;
  const std::string& as_string() const;
  const array_type& as_array() const;
  const object_type& as_object() const;

  const JsonValue& at(const std::string& key) const;
  std::string dump(int indent = 0) const;

 private:
  value_type value_;
};

class JsonParser {
 public:
  explicit JsonParser(std::string source);
  JsonValue parse();

 private:
  std::string source_;
  const char* data_;
  std::size_t size_;
  std::size_t pos_;

  void skip_whitespace();
  char peek() const;
  char consume();
  bool consume_if(char expected);
  JsonValue parse_value();
  JsonValue parse_object();
  JsonValue parse_array();
  JsonValue parse_number();
  JsonValue parse_string();
  JsonValue parse_literal(const char* literal, JsonValue value);
  [[noreturn]] void raise(const std::string& message) const;
};

JsonValue parse_json(std::string source);

}  // namespace executorch::runtime::core
