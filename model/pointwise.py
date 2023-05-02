class PositionwiseFeedForward(nn.Module):
    """
    A fully connected feed-forward network, which is applied to each position separately and identically.
    This consists of two linear transformations with a ReLU activation in between.
    The first layer takes inputs in model dimension and outputs in model dimension * width_ffn.
    and vice versa for the second layer
    """

    def __init__(self, d_model, width_ffn=4, dropout=0.1):
        super().__init__()
        self.linear1  = nn.Linear(d_model, d_model*width_ffn)
        self.linear2 = nn.Linear(width_ffn*d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(self.linear1(x).relu()))