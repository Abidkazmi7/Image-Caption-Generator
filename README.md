flowchart TD
    subgraph IMG["Image branch"]
        direction TB
        I1["Features_Input<br/>2048-d InceptionV3 vector"] --> I2["BatchNormalization"] --> I3["Dense 256<br/>ReLU"] --> I4["BatchNormalization<br/>256-d image encoding"]
    end

    subgraph TXT["Text branch"]
        direction TB
        T1["Sequence_Input<br/>padded token IDs"] --> T2["Embedding 256<br/>mask_zero = true"] --> T3["LSTM 256<br/>256-d text encoding"]
    end

    I4 --> M["add<br/>image + text, 256-d"]
    T3 --> M
    M --> D["Dense 256<br/>ReLU"] --> O["Output_Layer<br/>softmax over vocab_size"]

    class I1,I2,I3,I4 img
    class T1,T2,T3 txt
    class M,D,O merge