from ctu.parsers.base_parser import CTUParser

class CTUTrainParser(CTUParser):
  def _initialize(self, parser):

    CTUParser._initialize(self, parser)
        
    # TODO (shiyu) having both opt.mode and self.is_train is redundant?
    parser.set_defaults(mode='train')
    self.is_train = True

    return parser


