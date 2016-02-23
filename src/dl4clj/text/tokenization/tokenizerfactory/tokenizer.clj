(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/text/tokenization/tokenizerfactory/Tokenizer.html"}
  dl4clj.text.tokenization.tokenizerfactory.tokenizer
  (:import [org.deeplearning4j.text.tokenization.tokenizer Tokenizer]))

(defn count-tokens
  "The number of tokens in the tokenizer"
  [^Tokenizer this]
  (.countTokens this))

(defn get-tokens
  "Returns a list of all the tokens"
  [^Tokenizer this]
  (.getTokens this))

(defn has-more-tokens
  "An iterator for tracking whether more tokens are left in the iterator not"
  [^Tokenizer this]
  (.hasMoreTokens this))

(defn next-token
  "The next token (word usually) in the string"
  [^Tokenizer this]
  (.nextToken this))

(defn set-token-pre-processor
  "Set the token pre process"
  [^Tokenizer this token-pre-processor]
  (.setTokenPreProcessor this token-pre-processor))
