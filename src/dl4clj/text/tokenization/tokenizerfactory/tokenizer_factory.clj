(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/text/tokenization/tokenizerfactory/TokenizerFactory.html"}
  dl4clj.text.tokenization.tokenizerfactory.tokenizer-factory
  (:import [org.deeplearning4j.text.tokenization.tokenizerfactory TokenizerFactory]))

(defn create 
  "Create a tokenizer based on an input stream"
  [^TokenizerFactory x is]
  (.create x (clojure.java.io/input-stream is)))

(defn create-from-string 
  "Create a tokenizer from string"
  [^TokenizerFactory x ^String s]
  (.create x s))

(defn set-token-pre-processor 
  "Sets a token pre processor to be used with every tokenizer" 
  [^TokenizerFactory x pp]
  (.setTokenPreProcessor x pp))
