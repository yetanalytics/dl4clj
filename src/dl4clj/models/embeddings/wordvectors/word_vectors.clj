(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/models/embeddings/wordvectors/WordVectors.html"}
  dl4clj.models.embeddings.wordvectors.word-vectors
  (:import [org.deeplearning4j.models.embeddings.wordvectors WordVectors]))

(defn accuracy
  "Accuracy based on questions which are a space separated list of strings where the first
word is the query word, the next 2 words are negative, and the last word is the
predicted word to be nearest"
  [^WordVectors this string]
  (.accuracy this string))

(defn get-word-vector
  "Get the word vector for a given matrix"
  [^WordVectors this string]
  (into [] (.getWordVector this string)))

(defn get-word-vector-matrix
  "Get the word vector for a given matrix"
  [^WordVectors this string]
  (.getWordVectorMatrix this string))

(defn get-word-vector-matrix-normalized
  "Returns the word vector divided by the norm2 of the array"
  [^WordVectors this string]
  (.getWordVectorMatrixNormalized this string))

(defn has-word
  "Returns true if the model has this word in the vocab"
  [^WordVectors this string]
  (.hasWord this string))

(defn index-of
  [^WordVectors this string]
  (.indexOf this string))

(defn lookup-table
  "Lookup table for the vectors"
  [^WordVectors this]
  (.lookupTable this))

(defn similarity
  "Returns the similarity of 2 words"
  [^WordVectors this string1 string2]
  (.similarity this string1 string2))

(defn similar-words-in-vocab-to
  "Find all words with a similar characters in the vocab"
  [^WordVectors this string ^Double accuracy]
  (.similarWordsInVocabTo this string ^Double accuracy))

(defn vocab
  "Vocab for the vectors"
  [^WordVectors this]
  (.vocab this))

(defn words-nearest
  "Words nearest based on positive and negative"
  ;; ([^WordVectors this words]
  ;;  (.wordsNearest this words))
  ([^WordVectors this positive negative top]
   (.wordsNearest this positive negative (int top))))

(defn words-nearest-sum
  ;; ([^WordVectors this words]
  ;;  (.wordsNearestSum this words))
  ([^WordVectors this positive negative top]
   (.wordsNearestSum this positive negative (int top))))
