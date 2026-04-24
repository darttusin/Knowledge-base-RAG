from pydantic import BaseModel, EmailStr, field_validator


class UserAuth(BaseModel):
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def validate_password_length(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        return v


class UserRegister(BaseModel):
    email: EmailStr
    password: str
    username: str

    @field_validator("password")
    @classmethod
    def validate_password_length(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        return v


class UserChanges(BaseModel):
    email: EmailStr | None = None
    username: str | None = None
    old_password: str | None = None
    new_password: str | None = None

    @field_validator("new_password")
    @classmethod
    def validate_new_password_length(cls, v: str | None) -> str | None:
        if v is not None and len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        return v


class User(BaseModel):
    user_id: int
    email: EmailStr
    username: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str
    user: User


class ErrorMessage(BaseModel):
    detail: str
