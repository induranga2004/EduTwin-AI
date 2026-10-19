import { User } from '/imports/ui/Types/user';

interface TypingUser {
  user: User;
  userId: string;
}

export interface PublicTypingData {
  user_typing_public: TypingUser[];
}

export interface PrivateTypingData {
  user_typing_private: TypingUser[];
}

export type TypingData = PublicTypingData | PrivateTypingData;
