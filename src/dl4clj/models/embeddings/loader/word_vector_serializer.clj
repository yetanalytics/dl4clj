(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/models/embeddings/loader/WordVectorSerializer.html"}
  dl4clj.models.embeddings.loader.word-vector-serializer
  (:import [org.deeplearning4j.models.embeddings.loader WordVectorSerializer]
           [org.deeplearning4j.models.word2vec Word2Vec]
           [org.nd4j.linalg.api.ndarray INDArray]))

(defn word-vector-serializer []
  (WordVectorSerializer.))

(defn write-full-model
  "Saves full Word2Vec model in the way, that allows model updates without being rebuilt from scratches"
  [^Word2Vec vec ^String path]
  (WordVectorSerializer/writeFullModel vec path))

(defn write-tsne-format
  "Write the tsne format"
  [vec ^INDArray tsne csv]
  (WordVectorSerializer/writeTsneFormat vec tsne (clojure.java.io/as-file csv)))

(defn write-word-vectors
  "Writes the word vectors to the given path."
  ([^Word2Vec vec destination]
   (WordVectorSerializer/writeWordVectors vec ^java.io.BufferedWriter (clojure.java.io/writer destination))) 
  ([lookup-table cache path]
   (WordVectorSerializer/writeWordVectors lookup-table cache path)))

(defn load-full-model 
  [path]
  (WordVectorSerializer/loadFullModel path))

(defn load-google-model
  "Loads the google model"
  ([model-file binary?]
   (WordVectorSerializer/loadGoogleModel (clojure.java.io/as-file model-file) (boolean binary?)))
  ([model-file binary? line-breaks?]
   (WordVectorSerializer/loadGoogleModel (clojure.java.io/as-file model-file) (boolean binary?) (boolean line-breaks?))))

(defn load-txt 
  "Loads an in memory cache from the given path (sets syn0 and the vocab)"
  [f]
  (WordVectorSerializer/loadTxt (clojure.java.io/as-file f)))

(defn load-txt-vectors [f]
  "Loads an in memory cache from the given path (sets syn0 and the vocab)"
  (WordVectorSerializer/loadTxtVectors (clojure.java.io/as-file f)))
