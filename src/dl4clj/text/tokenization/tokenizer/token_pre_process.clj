#_(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/text/tokenization/tokenizer/TokenPreProcess.html"}
  dl4clj.text.tokenization.tokenizer.token-pre-process
  (:import [org.deeplearning4j.text.tokenization.tokenizer TokenPreProcess]))

#_(defn pre-process
  "Pre process a token"
  [^TokenPreProcess x ^String token]
  (.preProcess x token))

#_(deftype CustomTokenPreProcessor [preprocessor]
  TokenPreProcess
  (preProcess [this x] (preprocessor x)))

#_(defn token-pre-process [processor]
  (CustomTokenPreProcessor. processor))
